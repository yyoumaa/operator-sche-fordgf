"""
3302413python端作为决策大脑，做这几件事情：
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

MAX_REGIONS = 16
INIT_REGIONS = 1
NUM_FAMILY = 5
MAX_BATCH_FEEDBACKS = 64
MAX_ZERO_PER_FAMILY = 1

# 固定 region 共享内存布局偏移 (与 C 侧 __attribute__((packed)) 对齐)
# global_ctx(3*8=24) + regions_ctx(16*4*8=512) + num_feedbacks(4)
# + feedbacks(64*16=1024) + family_trials(5*4=20) = 1584
FEEDBACK_OFFSET = 536         # 24 + 512, 指向 num_feedbacks
TRIALS_OFFSET = 1564          # 536 + 4 + 1024, 指向 family_trials



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

class LinTSScheduler:
    FAMILY_PRIOR_STRENGTH = 1000.0

    def __init__(self, num_regions=MAX_REGIONS, num_families=5,
                 v=0.5, lambda_reg=1.0, temperature=1.0,
                 forgetting=1.0, epsilon=0.0):
        self.num_regions = num_regions
        self.num_families = num_families
        self.v = v                          # 后验方差缩放，控制探索幅度
        self.lambda_reg = lambda_reg        # ridge 正则
        self.temperature = temperature      # softmax 温度
        self.forgetting = forgetting        # 1.0=无遗忘, <1.0=遗忘老数据
        self.epsilon = epsilon              # epsilon-greedy 保底探索

        self.d_global = 3
        self.d_region_local = 4
        self.d = self.d_global + self.d_region_local   # 7
        self.d_fam = self.d_global                     # family层只用3维global

        # Family 层 (disjoint, 每个 family 一套)
        self.A_fam = np.stack([lambda_reg * np.eye(self.d_fam)
                               for _ in range(num_families)])   # (F, 3, 3)
        self.b_fam = np.zeros((num_families, self.d_fam))       # (F, 3)

        # Region 层 (条件于 family, 共享参数 over regions)
        self.A_reg = np.stack([lambda_reg * np.eye(self.d)
                               for _ in range(num_families)])   # (F, 7, 7)
        self.b_reg = np.zeros((num_families, self.d))           # (F, 7)

        self.family_reward_sum = np.zeros(num_families, dtype=np.float64)
        self.family_trial_sum = np.zeros(num_families, dtype=np.float64)
        self.update_count = 0

    def _softmax(self, logits, mask=None):
        logits = np.asarray(logits, dtype=np.float64)
        if mask is not None:
            logits = np.where(mask, logits, -1e9)
        logits = logits / max(self.temperature, 1e-6)
        logits -= np.max(logits)                 # 数值稳定
        exp = np.exp(logits)
        if mask is not None:
            exp = exp * mask
        s = exp.sum()
        if s < 1e-12:
            if mask is not None:
                n = max(int(mask.sum()), 1)
                return mask.astype(float) / n
            return np.ones_like(exp) / len(exp)
        return exp / s

    @staticmethod
    def _safe_cov(A_inv, scale):
        cov = scale * A_inv
        cov = 0.5 * (cov + cov.T)
        cov += 1e-9 * np.eye(cov.shape[0])
        return cov

    def get_distributions(self, global_ctx, regions_ctx_list, num_regions):
        valid_mask = np.array([np.any(np.abs(regions_ctx_list[i]) > 1e-12)
                               for i in range(num_regions)], dtype=bool)

        x_fam = np.asarray(global_ctx, dtype=np.float64)

        x_all = [None] * num_regions
        for i in range(num_regions):
            if valid_mask[i]:
                x_all[i] = np.concatenate(
                    [global_ctx, regions_ctx_list[i]]).astype(np.float64)

        family_bonus = self.get_family_trial_bonus()

        # ---- Family: Thompson Sampling ----
        fam_scores = np.zeros(self.num_families)
        for f in range(self.num_families):
            A_inv = np.linalg.inv(self.A_fam[f])
            theta_hat = A_inv @ self.b_fam[f]
            cov = self._safe_cov(A_inv, self.v**2)
            theta_sample = np.random.multivariate_normal(theta_hat, cov)
            fam_scores[f] = theta_sample @ x_fam + family_bonus[f]
        P_fam = self._softmax(fam_scores)

        # ---- Region: Thompson Sampling (每个 family 一套权重) ----
        P_reg_given_fam = np.zeros((self.num_families, num_regions))
        for f in range(self.num_families):
            A_inv = np.linalg.inv(self.A_reg[f])
            theta_hat = A_inv @ self.b_reg[f]
            cov = self._safe_cov(A_inv, self.v**2)
            theta_sample = np.random.multivariate_normal(theta_hat, cov)
            reg_scores = np.full(num_regions, -1e9)
            for i in range(num_regions):
                if valid_mask[i]:
                    reg_scores[i] = theta_sample @ x_all[i]
            P_reg_given_fam[f] = self._softmax(reg_scores, valid_mask)

        return P_fam, P_reg_given_fam, valid_mask, x_fam, x_all

    def apply_epsilon(self, P_fam, P_reg_given_fam, valid_mask):
        """epsilon-greedy 保底探索：以 epsilon 概率混入均匀分布"""
        eps = self.epsilon
        if eps <= 0:
            return P_fam, P_reg_given_fam

        # family: 混入均匀
        uniform_fam = np.ones(self.num_families) / self.num_families
        P_fam = (1 - eps) * P_fam + eps * uniform_fam

        # region: 混入均匀 (只在 valid region 上)
        n_valid = max(int(valid_mask.sum()), 1)
        uniform_reg = (valid_mask.astype(float) / n_valid)
        for f in range(self.num_families):
            P_reg_given_fam[f] = (1 - eps) * P_reg_given_fam[f] + eps * uniform_reg

        return P_fam, P_reg_given_fam

    def get_family_trial_bonus(self):
        avg_reward = self.family_reward_sum / np.maximum(self.family_trial_sum, 1.0)
        mean_avg_reward = float(np.mean(avg_reward))
        return self.FAMILY_PRIOR_STRENGTH * (avg_reward - mean_avg_reward)

    def update_region(self, best_family, best_region, x_all, reward):
        """单条正反馈更新 region posterior。"""
        if best_family == -1 or best_region == -1:
            return
        if not (0 <= best_family < self.num_families):
            return
        if not (0 <= best_region < self.num_regions) or x_all[best_region] is None:
            return

        gamma = self.forgetting
        x_r = x_all[best_region]
        self.A_reg[best_family] = (gamma * self.A_reg[best_family]
                                   + np.outer(x_r, x_r))
        self.b_reg[best_family] = (gamma * self.b_reg[best_family]
                                   + reward * x_r)

    def update_family_model(self, x_fam, feedbacks, family_trials):
        """每 batch 聚合更新 family posterior，每个 family 最多 1 次 update。
        reward = 该 family 本 batch 的 total_reward / positive_count，
        只除以有正反馈的次数，避免大量零 trial 稀释信号。"""
        gamma = self.forgetting

        fam_reward = np.zeros(self.num_families)
        fam_positive_count = np.zeros(self.num_families, dtype=int)
        for reward, _, family in feedbacks:
            if 0 <= family < self.num_families:
                fam_reward[family] += reward
                fam_positive_count[family] += 1

        updated = []
        for f in range(self.num_families):
            trials = max(int(family_trials[f]), 0) if f < len(family_trials) else 0
            if trials <= 0:
                continue
            pos_count = fam_positive_count[f]
            if pos_count <= 0:
                avg_r = 0.0
            else:
                avg_r = fam_reward[f] / pos_count
            self.A_fam[f] = gamma * self.A_fam[f] + np.outer(x_fam, x_fam)
            self.b_fam[f] = gamma * self.b_fam[f] + avg_r * x_fam
            updated.append((f, avg_r, pos_count, trials))

        self.update_count += 1
        if updated:
            parts = [f"f{f}:{avg_r:.4f}({p}pos/{t}t)" for f, avg_r, p, t in updated]
            logging.info(f"[PY][FAM_UPDATE] {' '.join(parts)} (t={self.update_count})")


class RegionManager:
    """自适应 region 分裂/合并管理器"""
    EVAL_INTERVAL = 50       # 每 50 个 batch 评估一次
    MIN_REGION_BYTES = 8     # region 最小字节数，低于此不分裂；避免 1-4 字节碎片化
    SPLIT_VAR_THRESH = 0.02  # reward 方差阈值，高于此考虑分裂
    MERGE_DIFF_THRESH = 0.01 # 相邻 region reward 均值差阈值，低于此考虑合并
    HISTORY_WINDOW = 50      # reward 历史窗口大小

    def __init__(self, max_regions=MAX_REGIONS, init_regions=INIT_REGIONS):
        self.max_regions = max_regions
        self.num_regions = init_regions
        self.bounds = []
        self.reward_history = [[] for _ in range(max_regions)]
        self.batch_count = 0

    def init_bounds(self, seed_len):
        """根据 seed 长度均匀初始化 bounds"""
        self.bounds = []
        base = seed_len // self.num_regions
        if base == 0:
            base = 1
        for i in range(self.num_regions):
            s = i * base
            e = seed_len if (i == self.num_regions - 1) else (i + 1) * base
            self.bounds.append((s, e))

    def set_from_shm(self, num_regions, bounds):
        """从共享内存读取的 bounds 同步到 manager"""
        self.num_regions = num_regions
        self.bounds = list(bounds[:num_regions])
        while len(self.reward_history) < self.max_regions:
            self.reward_history.append([])

    def record_rewards(self, feedbacks, num_regions):
        """记录每个 region 的 reward 到历史窗口"""
        for reward, region, family in feedbacks:
            if 0 <= region < num_regions:
                hist = self.reward_history[region]
                hist.append(reward)
                if len(hist) > self.HISTORY_WINDOW:
                    hist.pop(0)

    def maybe_adapt(self):
        """每 EVAL_INTERVAL 个 batch 评估分裂/合并，返回是否发生了变化"""
        self.batch_count += 1
        if self.batch_count % self.EVAL_INTERVAL != 0:
            return False

        changed = False
        changed |= self._try_split()
        changed |= self._try_merge()

        if changed:
            logging.info(f"[REGION][ADAPT] batch={self.batch_count} "
                         f"num_regions={self.num_regions} "
                         f"bounds={self.bounds}")
        return changed

    def _try_split(self):
        """找 reward 方差最大的 region 分裂。
        阈值随粒度自适应：粗分区天然方差低（空间平均效应），
        按 num_regions/max_regions 缩放，保证粗粒度时充分探索。
        """
        if self.num_regions >= self.max_regions:
            return False

        adaptive_thresh = self.SPLIT_VAR_THRESH * (self.num_regions / self.max_regions)

        best_idx = -1
        best_var = 0.0
        for i in range(self.num_regions):
            hist = self.reward_history[i]
            if len(hist) < 5:
                continue
            r_start, r_end = self.bounds[i]
            if (r_end - r_start) < self.MIN_REGION_BYTES * 2:
                continue
            var = np.var(hist)
            if var > adaptive_thresh and var > best_var:
                best_var = var
                best_idx = i

        if best_idx < 0:
            return False

        s, e = self.bounds[best_idx]
        mid = (s + e) // 2

        self.bounds[best_idx] = (s, mid)
        self.bounds.insert(best_idx + 1, (mid, e))

        self.reward_history[best_idx] = []
        self.reward_history.insert(best_idx + 1, [])

        self.num_regions += 1
        logging.info(f"[REGION][SPLIT] idx={best_idx} var={best_var:.4f} "
                     f"thresh={adaptive_thresh:.4f} "
                     f"[{s},{e}) -> [{s},{mid}) + [{mid},{e})")
        return True

    def _try_merge(self):
        """找 reward 均值最接近的相邻 region 合并"""
        if self.num_regions <= INIT_REGIONS:
            return False

        best_pair = -1
        best_diff = float('inf')
        for i in range(self.num_regions - 1):
            h1 = self.reward_history[i]
            h2 = self.reward_history[i + 1]
            if len(h1) < 5 or len(h2) < 5:
                continue
            m1, m2 = np.mean(h1), np.mean(h2)
            v1, v2 = np.var(h1), np.var(h2)
            diff = abs(m1 - m2)
            if diff < self.MERGE_DIFF_THRESH and v1 < self.SPLIT_VAR_THRESH \
                    and v2 < self.SPLIT_VAR_THRESH and diff < best_diff:
                best_diff = diff
                best_pair = i

        if best_pair < 0:
            return False

        i = best_pair
        s1, _ = self.bounds[i]
        _, e2 = self.bounds[i + 1]
        merged_hist = self.reward_history[i] + self.reward_history[i + 1]
        if len(merged_hist) > self.HISTORY_WINDOW:
            merged_hist = merged_hist[-self.HISTORY_WINDOW:]

        self.bounds[i] = (s1, e2)
        del self.bounds[i + 1]

        self.reward_history[i] = merged_hist
        del self.reward_history[i + 1]

        self.num_regions -= 1
        logging.info(f"[REGION][MERGE] idx={i},{i+1} diff={best_diff:.4f} "
                     f"-> [{s1},{e2})")
        return True



def init_ipc():
    global shm_c2py, shm_py2c, c2py_map, py2c_map
    global sem_c_done_features, sem_c_done_batch, sem_py_done_decision

    # 假设这些资源已经在 C 语言侧创建好了
    try:
        shm_c2py = posix_ipc.SharedMemory("/shm_c2py")
        c2py_map = mmap.mmap(shm_c2py.fd, 2048) # 足够容纳特征+多条反馈

        shm_py2c = posix_ipc.SharedMemory("/shm_py2c")
        py2c_map = mmap.mmap(shm_py2c.fd, 2048)  # 足够容纳决策

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


def write_decision_to_shm(py2c_map, P_fam: np.ndarray,
                          P_reg_given_fam: np.ndarray, num_regions: int,
                          bounds: list):
    """
    布局(family → region → bounds):
    [0:40]    P_fam             : 5 个 double
    [40:680]  P_reg_given_fam   : 5×MAX_REGIONS 个 double, 按 family 行主序
    [680:684] num_regions       : 1 个 int
    [684:812] bounds[16][2]     : 16×2 个 int (start, end)
    """
    full_P_reg = np.zeros((NUM_FAMILY, MAX_REGIONS), dtype=np.float64)
    full_P_reg[:, :num_regions] = P_reg_given_fam

    py2c_map.seek(0)
    py2c_map.write(P_fam.astype('<f8').tobytes())
    py2c_map.write(full_P_reg.astype('<f8').flatten().tobytes())

    py2c_map.write(struct.pack('<i', num_regions))
    for s, e in bounds:
        py2c_map.write(struct.pack('<2i', s, e))
    for _ in range(MAX_REGIONS - len(bounds)):
        py2c_map.write(struct.pack('<2i', 0, 0))
    py2c_map.flush()

    logging.info(f"[PY][SEND] P_fam={np.round(P_fam, 3).tolist()}")
    best_f = int(np.argmax(P_fam))
    logging.info(f"[PY][SEND] best_f={best_f} num_regions={num_regions} "
                 f"bounds={bounds[:num_regions]} "
                 f"P_reg|f={best_f}={np.round(P_reg_given_fam[best_f], 3).tolist()}")
    

def wake_up_afl():
    #做完后，通知C
    sem_py_done_decision.release()
    logging.info("Python: notified C")


def read_features_from_shm() -> tuple:
    """读 global_ctx + regions_ctx，global_ctx[0] 是归一化的 seed_len"""
    c2py_map.seek(0)
    global_ctx = np.frombuffer(c2py_map.read(3 * 8), dtype='<f8').copy().astype(np.float32)
    seed_len = int(round(global_ctx[0] * 10000))

    all_region_data = np.frombuffer(c2py_map.read(MAX_REGIONS * 4 * 8), dtype='<f8').copy()
    regions_matrix = all_region_data.reshape(MAX_REGIONS, 4).astype(np.float32)
    regions_ctx_list = [regions_matrix[i] for i in range(MAX_REGIONS)]

    return seed_len, global_ctx, regions_ctx_list


def read_batch_feedbacks(c2py_map):
    """
    从共享内存读取多条反馈 + 每个 family 的 trial 计数。
    固定 region 布局: num_feedbacks 在偏移 536, family_trials 在 1564
    """
    c2py_map.seek(FEEDBACK_OFFSET)
    num_feedbacks = struct.unpack('<i', c2py_map.read(4))[0]
    feedbacks = []
    for _ in range(num_feedbacks):
        reward = struct.unpack('<d', c2py_map.read(8))[0]
        region = struct.unpack('<i', c2py_map.read(4))[0]
        family = struct.unpack('<i', c2py_map.read(4))[0]
        feedbacks.append((float(reward), int(region), int(family)))

    c2py_map.seek(TRIALS_OFFSET)
    family_trials = list(struct.unpack('<5i', c2py_map.read(20)))

    return feedbacks, family_trials


def update_family_trials(scheduler, family_trials, feedbacks):
    gamma = scheduler.forgetting
    scheduler.family_reward_sum *= gamma
    scheduler.family_trial_sum *= gamma

    if not family_trials:
        return

    success_count = [0] * scheduler.num_families
    reward_sum = [0.0] * scheduler.num_families
    for reward, _, family in feedbacks:
        if 0 <= family < scheduler.num_families:
            success_count[family] += 1
            reward_sum[family] += reward

    for family in range(scheduler.num_families):
        trials = max(int(family_trials[family]), 0)
        if trials <= 0:
            continue
        scheduler.family_trial_sum[family] += trials
        scheduler.family_reward_sum[family] += reward_sum[family]

def main():
    global c2py_map, py2c_map
    init_ipc()

    region_mgr = RegionManager(max_regions=MAX_REGIONS, init_regions=INIT_REGIONS)

    scheduler = LinTSScheduler(
        num_regions=INIT_REGIONS, num_families=5,
        v=0.05,
        lambda_reg=1.0,
        temperature=0.1,
        forgetting=1.0,
        epsilon=0.01,
    )

    logging.info("===== LinTS Scheduler (family→region, adaptive) 启动 =====")
    logging.info(f"[PY][CONFIG] v={scheduler.v} temp={scheduler.temperature} "
                 f"forget={scheduler.forgetting} eps={scheduler.epsilon} "
                 f"init_regions={INIT_REGIONS}")

    bounds_initialized = False

    while True:
        wait_for_afl_features()
        seed_len, global_ctx, regions_ctx_list = read_features_from_shm()

        if not bounds_initialized and seed_len > 0:
            region_mgr.init_bounds(seed_len)
            bounds_initialized = True
            logging.info(f"[PY][REGION] init bounds seed_len={seed_len} "
                         f"num_regions={region_mgr.num_regions} "
                         f"bounds={region_mgr.bounds}")

        num_regions = region_mgr.num_regions
        scheduler.num_regions = num_regions

        P_fam, P_reg_given_fam, valid_mask, x_fam, x_all = \
            scheduler.get_distributions(global_ctx, regions_ctx_list, num_regions)

        P_fam, P_reg_given_fam = scheduler.apply_epsilon(
            P_fam, P_reg_given_fam, valid_mask)

        write_decision_to_shm(py2c_map, P_fam, P_reg_given_fam,
                              num_regions, region_mgr.bounds)
        wake_up_afl()

        wait_for_afl_finish_batch()
        feedbacks, family_trials = read_batch_feedbacks(c2py_map)
        update_family_trials(scheduler, family_trials, feedbacks)

        region_mgr.record_rewards(feedbacks, num_regions)
        if region_mgr.maybe_adapt():
            logging.info(f"[PY][REGION] adapted num_regions={region_mgr.num_regions} "
                         f"bounds={region_mgr.bounds}")

        logging.info(f"[PY][BATCH_DONE] num_feedbacks={len(feedbacks)} "
                     f"num_regions={region_mgr.num_regions} family_trials={family_trials}")

        scheduler.update_family_model(x_fam, feedbacks, family_trials)
        for reward, region, family in feedbacks:
            scheduler.update_region(
                best_family=family,
                best_region=region,
                x_all=x_all,
                reward=reward,
            )

if __name__ == "__main__":
    main()
