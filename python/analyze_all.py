#!/usr/bin/env python3
"""
三合一分析脚本：Python调度器日志 + C侧bandit日志 + AFL进度日志
用法: python3 /tmp/analyze_all.py
"""
import re, sys, numpy as np
from collections import defaultdict

PY_LOG = "/operator-sche-fordgf.log"
BANDIT_LOG = "/operator-sche-fordgf/bandit_log.txt"
AFL_LOG = "/log-test"

def parse_list(s):
    return [float(x) for x in s.strip('[]').split(',')]

print("=" * 60)
print("1. AFL 进度 (case count 增长)")
print("=" * 60)
cases = []
with open(AFL_LOG) as f:
    for line in f:
        m = re.search(r'(\d+) total', line)
        if m:
            cases.append(int(m.group(1)))
if cases:
    print(f"   最终 case 数: {cases[-1]}")
    print(f"   增长轨迹: {' → '.join(str(c) for c in sorted(set(cases)))}")
else:
    print("   未找到 case 数据")

print()
print("=" * 60)
print("2. Python 调度器分析")
print("=" * 60)

# 2a. 基本统计
batch_count = 0
feedback_counts = []
family_trials_list = []
pfam_list = []
update_count = 0
update_by_family = defaultdict(int)
update_by_region = defaultdict(int)
rewards = []
avg_rewards = []

with open(PY_LOG) as f:
    for line in f:
        if '[PY][BATCH_DONE]' in line:
            batch_count += 1
            m = re.search(r'num_feedbacks=(\d+)', line)
            if m: feedback_counts.append(int(m.group(1)))
            m = re.search(r'family_trials=\[(.*?)\]', line)
            if m: family_trials_list.append(parse_list(m.group(1)))

        elif '[PY][SEND] P_fam=' in line:
            m = re.search(r'P_fam=\[(.*?)\]', line)
            if m: pfam_list.append(parse_list(m.group(1)))

        elif '[PY][UPDATE]' in line:
            update_count += 1
            m = re.search(r'reward=([\d.]+).*best_f=(\d+).*best_r=(\d+).*avg_reward=([\d.]+)', line)
            if m:
                rewards.append(float(m.group(1)))
                update_by_family[int(m.group(2))] += 1
                update_by_region[int(m.group(3))] += 1
                avg_rewards.append(float(m.group(4)))

# 运行时间
times = []
with open(PY_LOG) as f:
    for line in f:
        m = re.search(r'(\d{2}:\d{2}:\d{2})', line)
        if m:
            times.append(m.group(1))
            if len(times) >= 2:
                break
with open(PY_LOG) as f:
    for line in f:
        m = re.search(r'(\d{2}:\d{2}:\d{2})', line)
        if m: last_time = m.group(1)

print(f"   运行时间: {times[0] if times else '?'} → {last_time if 'last_time' in dir() else '?'}")
print(f"   总 batch 数: {batch_count}")
print(f"   有正反馈的 batch: {sum(1 for c in feedback_counts if c > 0)} ({sum(1 for c in feedback_counts if c > 0)*100//max(batch_count,1)}%)")
print(f"   总 UPDATE 数: {update_count}")

if rewards:
    print(f"   reward: avg={np.mean(rewards):.4f} max={np.max(rewards):.4f} median={np.median(rewards):.4f}")

# 2b. P_fam 趋势
print()
print("   --- P_fam 趋势 (前5 / 中间5 / 后5) ---")
if len(pfam_list) >= 15:
    for label, sl in [("前5", pfam_list[:5]), ("中间", pfam_list[len(pfam_list)//2-2:len(pfam_list)//2+3]), ("后5", pfam_list[-5:])]:
        avg = np.mean(sl, axis=0)
        print(f"   {label}: [{', '.join(f'{v:.3f}' for v in avg)}]")
else:
    for p in pfam_list[-5:]:
        print(f"   [{', '.join(f'{v:.3f}' for v in p)}]")

# 2c. P_fam 标准差（衡量是否学到了偏好）
if pfam_list:
    all_pfam = np.array(pfam_list)
    spread = np.mean(np.std(all_pfam, axis=1))
    print(f"   P_fam 平均 spread (std): {spread:.4f}  (>0.03 说明有学到偏好)")

# 2d. family update 分布
print()
print("   --- 各 family 收到的 update 次数 ---")
for f in sorted(update_by_family):
    print(f"   family {f}: {update_by_family[f]} ({update_by_family[f]*100//max(update_count,1)}%)")

# 2e. region update 分布
print()
print("   --- 各 region 收到的 update 次数 (top 5) ---")
sorted_regions = sorted(update_by_region.items(), key=lambda x: -x[1])
for r, cnt in sorted_regions[:5]:
    print(f"   region {r}: {cnt} ({cnt*100//max(update_count,1)}%)")

# 2f. family_trials 趋势
print()
print("   --- family_trials 趋势 (最后5个batch) ---")
for ft in family_trials_list[-5:]:
    print(f"   {[int(x) for x in ft]}")

# 2g. avg_reward 趋势
if avg_rewards:
    last50 = avg_rewards[-50:]
    nonzero = [x for x in last50 if x > 0]
    print()
    print(f"   --- avg_reward (最后50条update) ---")
    print(f"   非零比例: {len(nonzero)}/{len(last50)}")
    if nonzero:
        print(f"   非零均值: {np.mean(nonzero):.6f}")

print()
print("=" * 60)
print("3. C 侧 bandit 日志分析")
print("=" * 60)

batch_end_count = 0
feedback_counts_c = []
hrew_updates = 0
sample_families = defaultdict(int)
sample_regions = defaultdict(int)

with open(BANDIT_LOG) as f:
    for line in f:
        if '[BANDIT][BATCH_END]' in line:
            batch_end_count += 1
            m = re.search(r'feedback_count=(\d+)', line)
            if m: feedback_counts_c.append(int(m.group(1)))
        elif '[BANDIT][SAMPLE]' in line:
            m = re.search(r'sampled_f=(\d+)', line)
            if m: sample_families[int(m.group(1))] += 1
            m = re.search(r'sampled_r=(\d+)', line)
            if m: sample_regions[int(m.group(1))] += 1
        elif '[HREW_UPDATE]' in line:
            hrew_updates += 1

print(f"   C 侧 batch 数: {batch_end_count}")
print(f"   C 侧 HREW_UPDATE 数: {hrew_updates}")

if feedback_counts_c:
    pos = [c for c in feedback_counts_c if c > 0]
    print(f"   有正反馈的 batch: {len(pos)} ({len(pos)*100//max(batch_end_count,1)}%)")
    if pos:
        print(f"   正反馈 batch 平均 feedback 数: {np.mean(pos):.1f}")

if sample_families:
    print()
    print("   --- 采样 family 分布 (前4个trial的stacking步骤) ---")
    total_s = sum(sample_families.values())
    for f in sorted(sample_families):
        print(f"   family {f}: {sample_families[f]} ({sample_families[f]*100//max(total_s,1)}%)")

print()
print("=" * 60)
print("总结")
print("=" * 60)
if pfam_list:
    last_pfam = pfam_list[-1]
    max_f = np.argmax(last_pfam)
    min_f = np.argmin(last_pfam)
    print(f"   最终 P_fam: [{', '.join(f'{v:.3f}' for v in last_pfam)}]")
    print(f"   最偏好 family: {max_f} ({last_pfam[max_f]:.3f}), 最不偏好: {min_f} ({last_pfam[min_f]:.3f})")
    diff = last_pfam[max_f] - last_pfam[min_f]
    if diff < 0.03:
        print(f"   判断: P_fam 接近均匀 (差值{diff:.3f})，family 层学习效果弱")
    elif diff < 0.08:
        print(f"   判断: P_fam 有轻微偏好 (差值{diff:.3f})，family 层开始分化")
    else:
        print(f"   判断: P_fam 有明显偏好 (差值{diff:.3f})，family 层学习有效")
if cases:
    print(f"   最终 case 数: {cases[-1]}")
