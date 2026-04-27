"""
Python端决策大脑 (per-operator版):
    1. 等待AFL传输特征
    2. LinTS采样算子概率 + 每个算子下的region概率
    3. 写决策到共享内存，通知AFL
    4. 等AFL跑完batch，读反馈，更新模型
"""
import os
import sys
import mmap
import posix_ipc
import logging, time
import struct
from dataclasses import dataclass, field
from typing import List
import random
import numpy as np

MAX_REGIONS = 16
INIT_REGIONS = 1
NUM_OPS = 15
MAX_BATCH_FEEDBACKS = 64

FEEDBACK_OFFSET = 536         # 24 + 512 -> num_feedbacks
TRIALS_OFFSET = 1564          # 536 + 4 + 64*16 -> op_trials

c2py_map = None
py2c_map = None

logging.basicConfig(
    filename="/operator-sche-fordgf.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='w'
)
logging.info("Python程序启动")

class LinTSScheduler:
    OP_PRIOR_STRENGTH = 1000.0

    def __init__(self, num_regions=MAX_REGIONS, num_ops=NUM_OPS,
                 v=0.5, lambda_reg=1.0, temperature=1.0,
                 forgetting=1.0, epsilon=0.0):
        self.num_regions = num_regions
        self.num_ops = num_ops
        self.v = v
        self.lambda_reg = lambda_reg
        self.temperature = temperature
        self.forgetting = forgetting
        self.epsilon = epsilon

        self.d_global = 3
        self.d_region_local = 4
        self.d = self.d_global + self.d_region_local   # 7
        self.d_op = self.d_global                      # op层只用3维global

        self.A_op = np.stack([lambda_reg * np.eye(self.d_op)
                              for _ in range(num_ops)])       # (15, 3, 3)
        self.b_op = np.zeros((num_ops, self.d_op))            # (15, 3)

        self.A_reg = np.stack([lambda_reg * np.eye(self.d)
                               for _ in range(num_ops)])      # (15, 7, 7)
        self.b_reg = np.zeros((num_ops, self.d))              # (15, 7)

        self.op_reward_sum = np.zeros(num_ops, dtype=np.float64)
        self.op_trial_sum = np.zeros(num_ops, dtype=np.float64)
        self.update_count = 0

    def _softmax(self, logits, mask=None):
        logits = np.asarray(logits, dtype=np.float64)
        if mask is not None:
            logits = np.where(mask, logits, -1e9)
        logits = logits / max(self.temperature, 1e-6)
        logits -= np.max(logits)
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

        x_op = np.asarray(global_ctx, dtype=np.float64)

        x_all = [None] * num_regions
        for i in range(num_regions):
            if valid_mask[i]:
                x_all[i] = np.concatenate(
                    [global_ctx, regions_ctx_list[i]]).astype(np.float64)

        op_bonus = self.get_op_trial_bonus()

        op_scores = np.zeros(self.num_ops)
        for op in range(self.num_ops):
            A_inv = np.linalg.inv(self.A_op[op])
            theta_hat = A_inv @ self.b_op[op]
            cov = self._safe_cov(A_inv, self.v**2)
            theta_sample = np.random.multivariate_normal(theta_hat, cov)
            op_scores[op] = theta_sample @ x_op + op_bonus[op]
        P_op = self._softmax(op_scores)

        P_reg_given_op = np.zeros((self.num_ops, num_regions))
        for op in range(self.num_ops):
            A_inv = np.linalg.inv(self.A_reg[op])
            theta_hat = A_inv @ self.b_reg[op]
            cov = self._safe_cov(A_inv, self.v**2)
            theta_sample = np.random.multivariate_normal(theta_hat, cov)
            reg_scores = np.full(num_regions, -1e9)
            for i in range(num_regions):
                if valid_mask[i]:
                    reg_scores[i] = theta_sample @ x_all[i]
            P_reg_given_op[op] = self._softmax(reg_scores, valid_mask)

        return P_op, P_reg_given_op, valid_mask, x_op, x_all

    def apply_epsilon(self, P_op, P_reg_given_op, valid_mask):
        eps = self.epsilon
        if eps <= 0:
            return P_op, P_reg_given_op

        uniform_op = np.ones(self.num_ops) / self.num_ops
        P_op = (1 - eps) * P_op + eps * uniform_op

        n_valid = max(int(valid_mask.sum()), 1)
        uniform_reg = (valid_mask.astype(float) / n_valid)
        for op in range(self.num_ops):
            P_reg_given_op[op] = (1 - eps) * P_reg_given_op[op] + eps * uniform_reg

        return P_op, P_reg_given_op

    def get_op_trial_bonus(self):
        avg_reward = self.op_reward_sum / np.maximum(self.op_trial_sum, 1.0)
        mean_avg_reward = float(np.mean(avg_reward))
        return self.OP_PRIOR_STRENGTH * (avg_reward - mean_avg_reward)

    def update_region(self, best_op, best_region, x_all, reward):
        if best_op == -1 or best_region == -1:
            return
        if not (0 <= best_op < self.num_ops):
            return
        if not (0 <= best_region < self.num_regions) or x_all[best_region] is None:
            return

        gamma = self.forgetting
        x_r = x_all[best_region]
        self.A_reg[best_op] = gamma * self.A_reg[best_op] + np.outer(x_r, x_r)
        self.b_reg[best_op] = gamma * self.b_reg[best_op] + reward * x_r

    def update_op_model(self, x_op, feedbacks, op_trials):
        gamma = self.forgetting

        op_reward = np.zeros(self.num_ops)
        op_positive_count = np.zeros(self.num_ops, dtype=int)
        for reward, _, op in feedbacks:
            if 0 <= op < self.num_ops:
                op_reward[op] += reward
                op_positive_count[op] += 1

        updated = []
        for op in range(self.num_ops):
            trials = max(int(op_trials[op]), 0) if op < len(op_trials) else 0
            if trials <= 0:
                continue
            pos_count = op_positive_count[op]
            if pos_count <= 0:
                avg_r = 0.0
            else:
                avg_r = op_reward[op] / pos_count
            self.A_op[op] = gamma * self.A_op[op] + np.outer(x_op, x_op)
            self.b_op[op] = gamma * self.b_op[op] + avg_r * x_op
            updated.append((op, avg_r, pos_count, trials))

        self.update_count += 1
        if updated:
            parts = [f"op{o}:{avg_r:.4f}({p}pos/{t}t)" for o, avg_r, p, t in updated]
            logging.info(f"[PY][OP_UPDATE] {' '.join(parts)} (t={self.update_count})")


class RegionManager:
    EVAL_INTERVAL = 50
    MIN_REGION_BYTES = 8
    SPLIT_VAR_THRESH = 0.02
    MERGE_DIFF_THRESH = 0.01
    HISTORY_WINDOW = 50

    def __init__(self, max_regions=MAX_REGIONS, init_regions=INIT_REGIONS):
        self.max_regions = max_regions
        self.num_regions = init_regions
        self.bounds = []
        self.reward_history = [[] for _ in range(max_regions)]
        self.batch_count = 0

    def init_bounds(self, seed_len):
        self.bounds = []
        base = seed_len // self.num_regions
        if base == 0:
            base = 1
        for i in range(self.num_regions):
            s = i * base
            e = seed_len if (i == self.num_regions - 1) else (i + 1) * base
            self.bounds.append((s, e))

    def set_from_shm(self, num_regions, bounds):
        self.num_regions = num_regions
        self.bounds = list(bounds[:num_regions])
        while len(self.reward_history) < self.max_regions:
            self.reward_history.append([])

    def record_rewards(self, feedbacks, num_regions):
        for reward, region, op in feedbacks:
            if 0 <= region < num_regions:
                hist = self.reward_history[region]
                hist.append(reward)
                if len(hist) > self.HISTORY_WINDOW:
                    hist.pop(0)

    def maybe_adapt(self):
        self.batch_count += 1
        if self.batch_count % self.EVAL_INTERVAL != 0:
            return False
        changed = False
        changed |= self._try_split()
        changed |= self._try_merge()
        if changed:
            logging.info(f"[REGION][ADAPT] batch={self.batch_count} "
                         f"num_regions={self.num_regions} bounds={self.bounds}")
        return changed

    def _try_split(self):
        if self.num_regions >= self.max_regions:
            return False
        adaptive_thresh = self.SPLIT_VAR_THRESH * (self.num_regions / self.max_regions)
        best_idx, best_var = -1, 0.0
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
                     f"thresh={adaptive_thresh:.4f} [{s},{e}) -> [{s},{mid}) + [{mid},{e})")
        return True

    def _try_merge(self):
        if self.num_regions <= INIT_REGIONS:
            return False
        best_pair, best_diff = -1, float('inf')
        for i in range(self.num_regions - 1):
            h1, h2 = self.reward_history[i], self.reward_history[i + 1]
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
        logging.info(f"[REGION][MERGE] idx={i},{i+1} diff={best_diff:.4f} -> [{s1},{e2})")
        return True


def init_ipc():
    global shm_c2py, shm_py2c, c2py_map, py2c_map
    global sem_c_done_features, sem_c_done_batch, sem_py_done_decision

    try:
        shm_c2py = posix_ipc.SharedMemory("/shm_c2py")
        c2py_map = mmap.mmap(shm_c2py.fd, 2048)

        shm_py2c = posix_ipc.SharedMemory("/shm_py2c")
        py2c_map = mmap.mmap(shm_py2c.fd, 4096)

        sem_c_done_features = posix_ipc.Semaphore("/sem_c_feat")
        sem_c_done_batch = posix_ipc.Semaphore("/sem_c_batch")
        sem_py_done_decision = posix_ipc.Semaphore("/sem_py_dec")

        logging.info("IPC resources linked successfully.")
    except Exception as e:
        logging.error(f"IPC Init Failed: {e}")
        sys.exit(1)

def wait_for_afl_finish_batch():
    try:
        sem_c_done_batch.acquire()
        logging.info("Python: Batch finished, reading reward...")
    except posix_ipc.SignalError:
        pass

def wait_for_afl_features():
    try:
        sem_c_done_features.acquire()
        logging.info("Python: New seed ready, reading features...")
    except posix_ipc.SignalError:
        pass


def write_decision_to_shm(py2c_map, P_op: np.ndarray,
                          P_reg_given_op: np.ndarray, num_regions: int,
                          bounds: list):
    """
    新布局 (op → region → bounds):
    [0:128]    P_op[16]              : 16 个 double (只用前15)
    [128:2176] P_reg_given_op[16][16]: 16×16 个 double
    [2176:2180] num_regions          : 1 个 int
    [2180:2308] bounds[16][2]        : 16×2 个 int
    """
    full_P_op = np.zeros(16, dtype=np.float64)
    full_P_op[:len(P_op)] = P_op

    full_P_reg = np.zeros((16, MAX_REGIONS), dtype=np.float64)
    full_P_reg[:P_reg_given_op.shape[0], :P_reg_given_op.shape[1]] = P_reg_given_op

    py2c_map.seek(0)
    py2c_map.write(full_P_op.astype('<f8').tobytes())
    py2c_map.write(full_P_reg.astype('<f8').flatten().tobytes())
    py2c_map.write(struct.pack('<i', num_regions))
    for s, e in bounds:
        py2c_map.write(struct.pack('<2i', s, e))
    for _ in range(MAX_REGIONS - len(bounds)):
        py2c_map.write(struct.pack('<2i', 0, 0))
    py2c_map.flush()

    logging.info(f"[PY][SEND] P_op={np.round(P_op, 3).tolist()}")
    best_op = int(np.argmax(P_op))
    logging.info(f"[PY][SEND] best_op={best_op} num_regions={num_regions} "
                 f"bounds={bounds[:num_regions]} "
                 f"P_reg|op={best_op}={np.round(P_reg_given_op[best_op], 3).tolist()}")


def wake_up_afl():
    sem_py_done_decision.release()
    logging.info("Python: notified C")


def read_features_from_shm() -> tuple:
    c2py_map.seek(0)
    global_ctx = np.frombuffer(c2py_map.read(3 * 8), dtype='<f8').copy().astype(np.float32)
    seed_len = int(round(global_ctx[0] * 10000))
    all_region_data = np.frombuffer(c2py_map.read(MAX_REGIONS * 4 * 8), dtype='<f8').copy()
    regions_matrix = all_region_data.reshape(MAX_REGIONS, 4).astype(np.float32)
    regions_ctx_list = [regions_matrix[i] for i in range(MAX_REGIONS)]
    return seed_len, global_ctx, regions_ctx_list


def read_batch_feedbacks(c2py_map):
    c2py_map.seek(FEEDBACK_OFFSET)
    num_feedbacks = struct.unpack('<i', c2py_map.read(4))[0]
    feedbacks = []
    for _ in range(num_feedbacks):
        reward = struct.unpack('<d', c2py_map.read(8))[0]
        region = struct.unpack('<i', c2py_map.read(4))[0]
        op = struct.unpack('<i', c2py_map.read(4))[0]
        feedbacks.append((float(reward), int(region), int(op)))

    c2py_map.seek(TRIALS_OFFSET)
    op_trials = list(struct.unpack('<16i', c2py_map.read(64)))
    return feedbacks, op_trials


def update_op_trials(scheduler, op_trials, feedbacks):
    gamma = scheduler.forgetting
    scheduler.op_reward_sum *= gamma
    scheduler.op_trial_sum *= gamma

    if not op_trials:
        return

    reward_sum = [0.0] * scheduler.num_ops
    for reward, _, op in feedbacks:
        if 0 <= op < scheduler.num_ops:
            reward_sum[op] += reward

    for op in range(scheduler.num_ops):
        trials = max(int(op_trials[op]), 0)
        if trials <= 0:
            continue
        scheduler.op_trial_sum[op] += trials
        scheduler.op_reward_sum[op] += reward_sum[op]

def main():
    global c2py_map, py2c_map
    init_ipc()

    region_mgr = RegionManager(max_regions=MAX_REGIONS, init_regions=INIT_REGIONS)

    scheduler = LinTSScheduler(
        num_regions=INIT_REGIONS, num_ops=NUM_OPS,
        v=0.05,
        lambda_reg=1.0,
        temperature=0.1,
        forgetting=1.0,
        epsilon=0.01,
    )

    logging.info("===== LinTS Scheduler (per-operator, adaptive region) 启动 =====")
    logging.info(f"[PY][CONFIG] num_ops={NUM_OPS} v={scheduler.v} temp={scheduler.temperature} "
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

        P_op, P_reg_given_op, valid_mask, x_op, x_all = \
            scheduler.get_distributions(global_ctx, regions_ctx_list, num_regions)

        P_op, P_reg_given_op = scheduler.apply_epsilon(
            P_op, P_reg_given_op, valid_mask)

        write_decision_to_shm(py2c_map, P_op, P_reg_given_op,
                              num_regions, region_mgr.bounds)
        wake_up_afl()

        wait_for_afl_finish_batch()
        feedbacks, op_trials = read_batch_feedbacks(c2py_map)
        update_op_trials(scheduler, op_trials, feedbacks)

        region_mgr.record_rewards(feedbacks, num_regions)
        if region_mgr.maybe_adapt():
            logging.info(f"[PY][REGION] adapted num_regions={region_mgr.num_regions} "
                         f"bounds={region_mgr.bounds}")

        logging.info(f"[PY][BATCH_DONE] num_feedbacks={len(feedbacks)} "
                     f"num_regions={region_mgr.num_regions} op_trials={op_trials[:NUM_OPS]}")

        scheduler.update_op_model(x_op, feedbacks, op_trials)
        for reward, region, op in feedbacks:
            scheduler.update_region(
                best_op=op,
                best_region=region,
                x_all=x_all,
                reward=reward,
            )

if __name__ == "__main__":
    main()
