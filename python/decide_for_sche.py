"""
Python端决策大脑 (纯算子版):
    1. 等待AFL传输特征
    2. LinTS采样算子概率
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
    OP_PRIOR_STRENGTH = 120.0

    def __init__(self, num_ops=NUM_OPS,
                 v=0.5, lambda_reg=1.0, temperature=1.0,
                 forgetting=1.0, epsilon=0.0):
        self.num_ops = num_ops
        self.v = v
        self.lambda_reg = lambda_reg
        self.temperature = temperature
        self.forgetting = forgetting
        self.epsilon = epsilon

        self.d_op = 3

        self.A_op = np.stack([lambda_reg * np.eye(self.d_op)
                              for _ in range(num_ops)])       # (15, 3, 3)
        self.b_op = np.zeros((num_ops, self.d_op))            # (15, 3)

        self.op_reward_sum = np.zeros(num_ops, dtype=np.float64)
        self.op_trial_sum = np.zeros(num_ops, dtype=np.float64)
        self.update_count = 0

    def _softmax(self, logits):
        logits = np.asarray(logits, dtype=np.float64)
        logits = logits / max(self.temperature, 1e-6)
        logits -= np.max(logits)
        exp = np.exp(logits)
        s = exp.sum()
        if s < 1e-12:
            return np.ones_like(exp) / len(exp)
        return exp / s

    @staticmethod
    def _safe_cov(A_inv, scale):
        cov = scale * A_inv
        cov = 0.5 * (cov + cov.T)
        cov += 1e-9 * np.eye(cov.shape[0])
        return cov

    def get_op_distribution(self, global_ctx):
        x_op = np.asarray(global_ctx, dtype=np.float64)
        op_bonus = self.get_op_trial_bonus()

        op_scores = np.zeros(self.num_ops)
        for op in range(self.num_ops):
            A_inv = np.linalg.inv(self.A_op[op])
            theta_hat = A_inv @ self.b_op[op]
            cov = self._safe_cov(A_inv, self.v**2)
            theta_sample = np.random.multivariate_normal(theta_hat, cov)
            op_scores[op] = theta_sample @ x_op + op_bonus[op]
        P_op = self._softmax(op_scores)

        # epsilon-greedy mixing
        if self.epsilon > 0:
            uniform_op = np.ones(self.num_ops) / self.num_ops
            P_op = (1 - self.epsilon) * P_op + self.epsilon * uniform_op

        return P_op, x_op

    def get_op_trial_bonus(self):
        avg_reward = self.op_reward_sum / np.maximum(self.op_trial_sum, 1.0)
        mean_avg_reward = float(np.mean(avg_reward))
        return self.OP_PRIOR_STRENGTH * (avg_reward - mean_avg_reward)

    def update_op_model(self, x_op, feedbacks, op_trials):
        gamma = self.forgetting

        op_success_rate = np.zeros(self.num_ops)
        op_has_data = np.zeros(self.num_ops, dtype=bool)
        for success_rate, op in feedbacks:
            if 0 <= op < self.num_ops:
                op_success_rate[op] = success_rate
                op_has_data[op] = True

        updated = []
        for op in range(self.num_ops):
            if not op_has_data[op]:
                continue
            trials = max(int(op_trials[op]), 0) if op < len(op_trials) else 0
            r = op_success_rate[op]
            self.A_op[op] = gamma * self.A_op[op] + np.outer(x_op, x_op)
            self.b_op[op] = gamma * self.b_op[op] + r * x_op
            updated.append((op, r, trials))

        self.update_count += 1
        if updated:
            parts = [f"op{o}:sr={r:.4f}({t}t)" for o, r, t in updated]
            logging.info(f"[PY][OP_UPDATE] {' '.join(parts)} (t={self.update_count})")


def init_ipc():
    global shm_c2py, shm_py2c, c2py_map, py2c_map
    global sem_c_done_features, sem_c_done_batch, sem_py_done_decision

    try:
        shm_c2py = posix_ipc.SharedMemory("/shm_c2py")
        c2py_map = mmap.mmap(shm_c2py.fd, 2048)

        shm_py2c = posix_ipc.SharedMemory("/shm_py2c")
        py2c_map = mmap.mmap(shm_py2c.fd, 256)

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


def write_decision_to_shm(py2c_map, P_op: np.ndarray):
    """
    简化布局 (只有算子概率):
    [0:128]  P_op[16]: 16 个 double (只用前15)
    """
    full_P_op = np.zeros(16, dtype=np.float64)
    full_P_op[:len(P_op)] = P_op

    py2c_map.seek(0)
    py2c_map.write(full_P_op.astype('<f8').tobytes())
    py2c_map.flush()

    logging.info(f"[PY][SEND] P_op={np.round(P_op, 3).tolist()}")


def wake_up_afl():
    sem_py_done_decision.release()
    logging.info("Python: notified C")


def read_features_from_shm() -> tuple:
    c2py_map.seek(0)
    global_ctx = np.frombuffer(c2py_map.read(3 * 8), dtype='<f8').copy().astype(np.float32)
    return global_ctx


def read_batch_feedbacks(c2py_map):
    c2py_map.seek(FEEDBACK_OFFSET)
    num_feedbacks = struct.unpack('<i', c2py_map.read(4))[0]
    feedbacks = []
    for _ in range(num_feedbacks):
        reward = struct.unpack('<d', c2py_map.read(8))[0]
        _region = struct.unpack('<i', c2py_map.read(4))[0]  # ignored
        op = struct.unpack('<i', c2py_map.read(4))[0]
        feedbacks.append((float(reward), int(op)))

    c2py_map.seek(TRIALS_OFFSET)
    op_trials = list(struct.unpack('<16i', c2py_map.read(64)))
    return feedbacks, op_trials


def update_op_trials(scheduler, op_trials, feedbacks):
    gamma = scheduler.forgetting
    scheduler.op_reward_sum *= gamma
    scheduler.op_trial_sum *= gamma

    if not op_trials:
        return

    for success_rate, op in feedbacks:
        if 0 <= op < scheduler.num_ops:
            trials = max(int(op_trials[op]), 0)
            if trials <= 0:
                continue
            scheduler.op_trial_sum[op] += 1
            scheduler.op_reward_sum[op] += success_rate

def main():
    global c2py_map, py2c_map
    init_ipc()

    scheduler = LinTSScheduler(
        num_ops=NUM_OPS,
        v=0.05,
        lambda_reg=1.0,
        temperature=0.4,
        forgetting=1.0,
        epsilon=0.02,
    )

    logging.info("===== LinTS Scheduler (pure operator, no region) 启动 =====")
    logging.info(f"[PY][CONFIG] num_ops={NUM_OPS} v={scheduler.v} temp={scheduler.temperature} "
                 f"forget={scheduler.forgetting} eps={scheduler.epsilon} "
                 f"STRENGTH={scheduler.OP_PRIOR_STRENGTH}")

    while True:
        wait_for_afl_features()
        global_ctx = read_features_from_shm()

        P_op, x_op = scheduler.get_op_distribution(global_ctx)

        write_decision_to_shm(py2c_map, P_op)
        wake_up_afl()

        wait_for_afl_finish_batch()
        feedbacks, op_trials = read_batch_feedbacks(c2py_map)
        update_op_trials(scheduler, op_trials, feedbacks)

        logging.info(f"[PY][BATCH_DONE] num_feedbacks={len(feedbacks)} op_trials={op_trials[:NUM_OPS]}")

        scheduler.update_op_model(x_op, feedbacks, op_trials)

if __name__ == "__main__":
    main()
