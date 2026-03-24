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
import logging,time
import struct
from dataclasses import dataclass, field
from typing import List
import random
import numpy as np

NUM_REGIONS=8
NUM_FAMILY = 6
ALPHA_DEFAULT=1.0
ALPHA_TRUE=1.5
# 常量定义，避免在循环里计算
# 66 个 double，每个 8 字节，总计 528 字节
FEATURE_BYTES_LEN = (2 + 16 * 4) * 8



#声明全局信号量和共享内存
c2py_map = None
py2c_map = None


logging.basicConfig(
    filename="/operator-sche-fordgf.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Python程序启动")


class LinUCBArm:
    def __init__(self, d: int, alpha: float = ALPHA_DEFAULT):
        self.d = d                             # 上下文特征的维度
        self.alpha = alpha                     # 探索权重参数
        self.A = np.identity(d)                # d x d 矩阵
        self.b = np.zeros(d)                   # d 维向量
        self.A_inv = np.identity(d)

    def get_ucb(self, context: np.ndarray) -> float:
        # context 必须是 shape 为 (d,) 的 numpy 数组
        theta = self.A_inv.dot(self.b)         # 估计参数 theta 
        expected_reward = theta.dot(context)
        uncertainty = self.alpha * np.sqrt(context.dot(self.A_inv).dot(context))
        return expected_reward + uncertainty

    def update(self, context: np.ndarray, reward: float):
        # 严格执行: 遇到 reward 联合更新矩阵 
        self.A += np.outer(context, context)
        self.b += reward * context
        self.A_inv = np.linalg.inv(self.A)



class HierarchicalBandit:
    def __init__(self, num_regions=16, num_families=6, alpha=ALPHA_DEFAULT):
        # 定义特征维度
        self.dim_region_feat = 4   # 局部特征维度
        self.dim_global_feat = 2   # 全局特征维度
        self.dim_combined = self.dim_region_feat + self.dim_global_feat  # 联合特征维度
        
        # 大脑 1：Region 选择器 (16 个 Arm，每个 Arm 看 4 维特征)
        # 创建一个list，里面每一项都是一个a区域的rm，不同arm之间不共享参数
        self.region_arms = [LinUCBArm(d=self.dim_region_feat, alpha=alpha) for _ in range(num_regions)]
        
        # 大脑 2：算子族选择器 (6 个 Arm，每个 Arm 看 6 维特征)
        self.family_arms = [LinUCBArm(d=self.dim_combined, alpha=alpha) for _ in range(num_families)]

    def get_action(self, global_context: np.ndarray, regions_context_list: list):
        """
        前向选择：先选哪块地 (Region)，再选哪把工具 (Family)
        :param global_context: shape 为 (2,) 的 np array
        :param regions_context_list: 包含 16 个元素的 list，每个元素是 shape 为 (4,) 的 np array
        """
        # ==========================================
        # 步骤 1: 大脑 1 选拔潜力最大的 Region
        # ==========================================
        best_region_idx = -1
        max_region_ucb = -float('inf')
        
        for idx, arm in enumerate(self.region_arms):
            region_ctx = regions_context_list[idx]
            ucb_val = arm.get_ucb(region_ctx)
            if ucb_val > max_region_ucb:
                max_region_ucb = ucb_val
                best_region_idx = idx
                
        # 保存选中的局部特征，作为下一步的垫脚石
        selected_region_context = regions_context_list[best_region_idx]

        # ==========================================
        # 步骤 2: 上下文单向传递与特征拼接
        # ==========================================
        # 大脑 2 踩在大脑 1 的肩膀上，拼接出 5 维联合特征
        combined_context = np.concatenate((selected_region_context, global_context))

        # ==========================================
        # 步骤 3: 大脑 2 选拔最合适的 Operator Family
        # ==========================================
        best_family_idx = -1
        max_family_ucb = -float('inf')
        
        for idx, arm in enumerate(self.family_arms):
            ucb_val = arm.get_ucb(combined_context)
            if ucb_val > max_family_ucb:
                max_family_ucb = ucb_val
                best_family_idx = idx

        # 返回决策结果，并附带本次决策使用的 Context (用于后续 Update)
        return best_region_idx, best_family_idx, selected_region_context, combined_context

    def update(self, region_idx: int, family_idx: int, 
               region_context: np.ndarray, combined_context: np.ndarray, 
               reward: float):
        """
        后向更新：共享同一份工资 (Reward) 强绑定
        """
        # 大脑 1 拿工资：更新选中的那块地的评价
        self.region_arms[region_idx].update(region_context, reward)
        
        # 大脑 2 拿同样一份工资：更新在这块地里用这把工具的评价
        self.family_arms[family_idx].update(combined_context, reward)


def init_ipc():
    global shm_c2py, shm_py2c, c2py_map, py2c_map
    global sem_c_done_features, sem_c_done_batch, sem_py_done_decision

    # 假设这些资源已经在 C 语言侧创建好了
    try:
        shm_c2py = posix_ipc.SharedMemory("/shm_c2py")
        c2py_map = mmap.mmap(shm_c2py.fd, 1024) # 足够容纳特征+Reward

        shm_py2c = posix_ipc.SharedMemory("/shm_py2c")
        py2c_map = mmap.mmap(shm_py2c.fd, 128)  # 足够容纳决策

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


def write_decision_to_shm(py2c_map, region_id: int, family_id: int):
    """
    将决策结果以二进制形式写入共享内存。
    格式：2 个连续的 4 字节整数 (Little-endian int)
    总计：8 字节
    """
    # 'ii' 表示两个 int，'<' 表示小端序（兼容大多数 Linux/X86 环境）
    binary_data = struct.pack('<ii', region_id, family_id)
    
    # 回到起始位置并写入
    py2c_map.seek(0)
    py2c_map.write(binary_data)
    
    # 刷新映射，确保 C 语言侧能立刻看到更新（在某些系统上 mmap 需要 flush）
    py2c_map.flush()
    
    logging.info(f"Python sent decision: Region {region_id}, Family {family_id}")

    

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
    global_ctx = data_array[0:2].astype(np.float32)
    
    # 4. 切片并重塑：后 64 个元素是 16 个 Region 的特征
    # 直接 reshape 成 (16, 4) 的二维矩阵
    regions_matrix = data_array[2:].reshape(16, 4).astype(np.float32)
    
    # 将矩阵转为 list 
    regions_ctx_list = [regions_matrix[i, :] for i in range(16)]
    
    return global_ctx, regions_ctx_list


def read_reward_from_shm(c2py_map) -> float:
    """
    从共享内存的固定偏移量处读取本次 Batch 的 Reward。
    假设 Reward 放在特征数据之后（偏移量 528 处）。
    """
    # 16*4*8 + 2*8 = 528
    REWARD_OFFSET = 528 
    
    c2py_map.seek(REWARD_OFFSET)
    # 读取 8 字节并解析为双精度浮点数 (double)
    raw_reward = c2py_map.read(8)
    reward = struct.unpack('<d', raw_reward)[0]
    
    return float(reward)


def main():
    global c2py_map, py2c_map
    init_ipc()
     # 1. 全局初始化 (只执行一次)
    bandit_manager = HierarchicalBandit(num_regions=16, num_families=6, alpha=ALPHA_TRUE)

    # --- 不断接收 AFL 的请求 ---
    while True:
        ## 等待 C 算完特征
        wait_for_afl_features()

        # 2. 从共享内存解析出特征
        global_ctx,regions_ctx_list = read_features_from_shm()        # 例如: np.array([0.8, -0.2])以及 长度 16 的 list
             
        # 3. 联合决策
        # 注意：我们要把做决策时用的 context 存下来，因为 C 语言跑完一个 Batch 之后，
        # 环境可能变化了，更新时必须用“当时做决策的 Context”。
        region_id, family_id, used_reg_ctx, used_comb_ctx = bandit_manager.get_action(global_ctx, regions_ctx_list)

        # 4. 把决策写回共享内存，放行 C 语言去跑 Fuzzing Batch
        write_decision_to_shm(py2c_map, region_id, family_id)
        wake_up_afl()
 
        # 5. 等待 C 语言跑完这n次，拿回最高/平均/... Reward 
        wait_for_afl_finish_batch()
        
        batch_reward = read_reward_from_shm(c2py_map)

        # 6. 强绑定联合更新
        bandit_manager.update(region_idx=region_id, 
                            family_idx=family_id, 
                            region_context=used_reg_ctx, 
                            combined_context=used_comb_ctx, 
                            reward=batch_reward)


  

if __name__=="__main__":
    main()
