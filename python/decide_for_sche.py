"""
python端作为决策大脑，做这几件事情：
    1. 等待AFL那边传输数据
    2. 保存数据到结构体，处理成特征
    3. 用这次的特征和上一次的reward更新决策，选择Region
    4. 用拼接后的特征和上次的reward做第二次决策，选择算子族
    5. 把决策写入共享内存，通知afl可以读取数据
    6. 用这次的决策作为action记录，为下一轮bandit更新做准备
"""
import os
import sys
import mmap
import posix_ipc
import logging,time5
import struct
from dataclasses import dataclass, field
from typing import List
import random
import numpy as np

NUM_REGIONS=16
NUM_FAMILY = 5
ALPHA_DEFAULT=1.0
ALPHA_TRUE=1.5
# 常量定义，避免在循环里计算
# 66 个 double，每个 8 字节，总计 536 字节
FEATURE_BYTES_LEN = (3 + 16 * 4) * 8



#声明全局信号量和共享内存
c2py_map = None
py2c_map = None


logging.basicConfig(
    filename="/operator-sche-fordgf.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='w'  # 加这一行，每次启动覆盖而不是追加
)
logging.info("Python程序启动")

class PolicyGradientScheduler:
    def __init__(self, num_regions=16, num_families=5, lr=0.02, 
                 baseline_alpha=0.1, temperature=5.0):
        self.num_regions = num_regions
        self.num_families = num_families
        self.lr = lr
        self.baseline_alpha = baseline_alpha  # EMA衰减系数
        self.temperature = temperature         # softmax温度，越大越均匀，越小越集中
        
        self.dim_local = 4    # 局部特征维度
        self.dim_global = 3   # 全局特征维度
        self.dim_combined = self.dim_local + self.dim_global  # 7维

        # W_reg: shape (7,) 一个共享的权重向量，对所有region打分
        # 物理含义：学"什么特征的region更好"，而不是"第几号region更好"
        self.W_reg = np.zeros(self.dim_combined)
        
        # W_fam: shape (5, 7) 每个family一行，学"什么特征的region适合这个family"
        self.W_fam = np.zeros((num_families, self.dim_combined))
        
        # EMA baseline，用于方差缩减
        self.baseline = 0.0
        self.update_count = 0

    def _softmax(self, logits):
        logits = logits / self.temperature
        logits = logits - np.max(logits)  # 数值稳定
        exp_logits = np.exp(logits)
        return exp_logits / (np.sum(exp_logits) + 1e-12)

    def get_distributions(self, global_ctx, regions_ctx_list):
        """
        返回：
        P_reg: shape (16,)    每个region被选中的概率
        P_fam: shape (16, 5)  每个region下各family的概率
        valid_mask: shape (16,) 哪些region是有效的
        """
        region_scores = np.full(self.num_regions, -1e9)
        valid_mask = np.zeros(self.num_regions, dtype=bool)
        x_all = []  # 每个region的特征向量，用于梯度计算

        for i, region_ctx in enumerate(regions_ctx_list):
            if np.all(np.abs(region_ctx) < 1e-12):
                x_all.append(None)
                continue
            x_i = np.concatenate([global_ctx, region_ctx])
            x_all.append(x_i)
            region_scores[i] = np.dot(self.W_reg, x_i)
            valid_mask[i] = True

        P_reg = self._softmax(region_scores)
        # 把无效region的概率清零并重新归一化
        P_reg = P_reg * valid_mask
        total = P_reg.sum()
        if total > 1e-12:
            P_reg = P_reg / total
        else:
            # 兜底：均匀分配给有效region
            n_valid = valid_mask.sum()
            P_reg = valid_mask.astype(float) / max(n_valid, 1)

        P_fam = np.zeros((self.num_regions, self.num_families))
        for i in range(self.num_regions):
            if not valid_mask[i] or x_all[i] is None:
                P_fam[i] = np.ones(self.num_families) / self.num_families
            else:
                family_logits = self.W_fam.dot(x_all[i])  # shape (5,)
                P_fam[i] = self._softmax(family_logits)

        return P_reg, P_fam, valid_mask, x_all

    def update(self, best_region, best_family, global_ctx, 
               regions_ctx_list, reward):
        """
        用C侧反馈的"最佳(region, family)"做REINFORCE更新。
        best_region, best_family: C侧batch中产生最高reward的那对动作
        reward: 这次batch的max_reward
        """
        # 新增：C侧传-1表示本批没有正收益，跳过更新
        if best_region == -1 or best_family == -1:
            logging.info(f"[PY][UPDATE] 本批无正收益，跳过权重更新")
            return

        # 更新EMA baseline
        self.baseline = ((1 - self.baseline_alpha) * self.baseline 
                         + self.baseline_alpha * reward)
        advantage = reward - self.baseline
        self.update_count += 1

        logging.info(f"[PY][UPDATE] reward={reward:.4f} baseline={self.baseline:.4f} "
                     f"advantage={advantage:.4f} best_r={best_region} best_f={best_family}")

        if abs(advantage) < 1e-9:
            return  # 和baseline持平，不更新

        # 重新计算当前分布（用于梯度）
        P_reg, P_fam, valid_mask, x_all = self.get_distributions(
            global_ctx, regions_ctx_list)

        # ---- 更新 W_reg ----
        # REINFORCE梯度（权重共享版本）：
        # grad = x_best_region - E_{p}[x]
        if valid_mask[best_region] and x_all[best_region] is not None:
            x_best = x_all[best_region]
            # 计算在当前策略下的期望特征向量
            expected_x = np.zeros(self.dim_combined)
            for i in range(self.num_regions):
                if valid_mask[i] and x_all[i] is not None:
                    expected_x += P_reg[i] * x_all[i]
            grad_W_reg = x_best - expected_x
            self.W_reg += self.lr * advantage * grad_W_reg

        # ---- 更新 W_fam ----
        # 对best_region对应的特征做更新
        if valid_mask[best_region] and x_all[best_region] is not None:
            x_r = x_all[best_region]
            p_fam = P_fam[best_region]
            for f in range(self.num_families):
                indicator = 1.0 if f == best_family else 0.0
                grad_f = (indicator - p_fam[f]) * x_r
                self.W_fam[f] += self.lr * advantage * grad_f

        # 防止权重爆炸，做一个轻量clip
        self.W_reg = np.clip(self.W_reg, -10.0, 10.0)
        self.W_fam = np.clip(self.W_fam, -10.0, 10.0)

def init_ipc():
    global shm_c2py, shm_py2c, c2py_map, py2c_map
    global sem_c_done_features, sem_c_done_batch, sem_py_done_decision

    # 假设这些资源已经在 C 语言侧创建好了
    try:
        shm_c2py = posix_ipc.SharedMemory("/shm_c2py")
        c2py_map = mmap.mmap(shm_c2py.fd, 1024) # 足够容纳特征+Reward

        shm_py2c = posix_ipc.SharedMemory("/shm_py2c")
        py2c_map = mmap.mmap(shm_py2c.fd, 1024)  # 足够容纳决策

        # 三个核心信号量
        sem_c_done_features = posix_ipc.Semaphore("/sem_c_feat")
        sem_c_done_batch = posix_ipc.Semaphore("/sem_c_batch")
        sem_py_done_decision = posix_ipc.Semaphore("/sem_py_dec")
        
        logging.info("IPC resources linked successfully.")
    except Exception as e:
        logging.error(f"IPC Init Failed: {e}")
        sys.exit(1)

def wait_for_afl_finish_batch():
    """
    专门等待 C 语言跑完 256 次变异并写回 Reward 的信号
    """
    # 移除 while True 和 time.sleep，posix_ipc 的 acquire 本身就是阻塞且高效的
    try:
        # 重命名信号量，使其语义明确：sem_c_done_batch
        sem_c_done_batch.acquire() 
        logging.info("Python: Batch finished, reading reward...")
    except posix_ipc.SignalError:
        # 处理被系统信号中断的情况
        pass

def wait_for_afl_features():
    """
    专门等待 C 语言准备好新种子的特征
    """
    try:
        sem_c_done_features.acquire()
        logging.info("Python: New seed ready, reading features...")
    except posix_ipc.SignalError:
        pass


def write_decision_to_shm(py2c_map, P_reg: np.ndarray, P_fam: np.ndarray):
    """
    将概率分布写入共享内存。
    格式：P_reg (16个double) + P_fam (16*5=80个double)
    总计：96个double = 768字节
    """
    py2c_map.seek(0)
    # P_reg: 16个float64
    py2c_map.write(P_reg.astype('<f8').tobytes())
    # P_fam: 80个float64（按行展平）
    py2c_map.write(P_fam.astype('<f8').flatten().tobytes())
    py2c_map.flush()

    logging.info(f"[PY][SEND] P_reg={np.round(P_reg,3).tolist()}")
    # 打印概率最高的那个region对应的family分布
    best_r = int(np.argmax(P_reg))
    logging.info(f"[PY][SEND] best_r={best_r} P_fam[{best_r}]={np.round(P_fam[best_r],3).tolist()}")

    

def wake_up_afl():
    #做完后，通知C
    sem_py_done_decision.release()
    logging.info("Python: notified C")


def read_features_from_shm() -> tuple:
    """
    使用二进制零拷贝方式，从共享内存瞬间映射出特征数组。
    """
    # 1. 直接读取固定的 528 字节
    c2py_map.seek(0)
    raw_bytes = c2py_map.read(FEATURE_BYTES_LEN)
    
    # 2. 核心：np.frombuffer
    # '<f8' 表示 Little-Endian (小端序) 的 64 位浮点数，完美对应 C 语言的 double
    # 这一步是零拷贝的，速度极快，直接将字节流映射为长度为 66 的一维数组
    data_array = np.frombuffer(raw_bytes, dtype='<f8')
    
    # 3. 切片提取：前 2 个元素是全局特征
    # .astype(np.float32) 是为了转成 32 位浮点数，加速后续 LinUCB 的矩阵相乘
    global_ctx = data_array[0:3].astype(np.float32)
    
    # 4. 切片并重塑：后 64 个元素是 16 个 Region 的特征
    # 直接 reshape 成 (16, 4) 的二维矩阵
    regions_matrix = data_array[3:].reshape(16, 4).astype(np.float32)
    
    # 将矩阵转为 list 
    regions_ctx_list = [regions_matrix[i, :] for i in range(16)]
    
    return global_ctx, regions_ctx_list


def read_reward_and_best_action(c2py_map):
    """
    从共享内存读取：
    - reward (1个double，偏移536)
    - best_region (1个int，偏移544)
    - best_family (1个int，偏移548)
    """
    REWARD_OFFSET = 536
    c2py_map.seek(REWARD_OFFSET)
    reward = struct.unpack('<d', c2py_map.read(8))[0]
    best_region = struct.unpack('<i', c2py_map.read(4))[0]
    best_family = struct.unpack('<i', c2py_map.read(4))[0]

    return float(reward), int(best_region), int(best_family)


def main():
    global c2py_map, py2c_map
    init_ipc()
    
    scheduler = PolicyGradientScheduler(
        num_regions=16, num_families=5,
        lr=0.005, baseline_alpha=0.1, temperature=3.0
    )
    logging.info("===== PolicyGradient Scheduler 启动 =====")


    while True:
        wait_for_afl_features()
        global_ctx, regions_ctx_list = read_features_from_shm()

        # 生成概率分布
        P_reg, P_fam, valid_mask, _ = scheduler.get_distributions(
            global_ctx, regions_ctx_list)

        # 发给C侧
        write_decision_to_shm(py2c_map, P_reg, P_fam)
        wake_up_afl()

        # 等待batch结束
        wait_for_afl_finish_batch()
        reward, best_region, best_family = read_reward_and_best_action(c2py_map)

        logging.info(f"[PY][BATCH_DONE] reward={reward:.4f} "
             f"best_r={best_region} best_f={best_family} "
             f"{'(no update)' if best_region == -1 else '(will update)'}")

        # 用这次的特征和C反馈的最佳动作做更新
        scheduler.update(
            best_region=best_region,
            best_family=best_family,
            global_ctx=global_ctx,
            regions_ctx_list=regions_ctx_list,
            reward=reward
        )

if __name__ == "__main__":
    main()
