"""
分析 /operator-sche-fordgf.log (Python scheduler 日志)
逐行解析，不全量加载。

适配自适应 region 新格式：
  - [PY][SEND] best_f=... num_regions=... P_reg|f=...=[...]
  - [PY][REGION_ADAPT] new num_regions=... bounds=...

提取:
  - P_fam 分布演化
  - P_reg|f 熵/区分度演化
  - num_regions 变化
  - split/merge 触发频率
  - UPDATE 统计
"""
import sys, re
import numpy as np
from collections import Counter
from datetime import datetime


def entropy(vals):
    arr = np.array(vals, dtype=np.float64)
    arr = arr[arr > 1e-12]
    if len(arr) == 0:
        return 0.0
    return float(-np.sum(arr * np.log(arr)))


def main(filepath):
    pfam_pat = re.compile(r'\[PY\]\[SEND\] P_fam=\[(.*?)\]')
    preg_pat = re.compile(r'\[PY\]\[SEND\] best_f=(\d+) num_regions=(\d+) P_reg\|f=\d+=\[(.*?)\]')
    batch_pat = re.compile(r'\[PY\]\[BATCH_DONE\] num_feedbacks=(\d+)')
    upd_pat = re.compile(r'\[PY\]\[UPDATE\] reward=([\d.]+) best_f=(\d+) best_r=(\d+) \(t=(\d+)\)')
    adapt_pat = re.compile(r'\[PY\]\[REGION_ADAPT\] new num_regions=(\d+) bounds=(.*)$')
    split_pat = re.compile(r'\[REGION\]\[SPLIT\]')
    merge_pat = re.compile(r'\[REGION\]\[MERGE\]')
    time_pat = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)')

    pfam_history = []
    preg_history = []
    feedback_counts = []
    updates = []
    timestamps = []
    batch_idx = 0
    region_hist = Counter()
    adapt_events = []
    split_count = 0
    merge_count = 0

    with open(filepath) as f:
        for line in f:
            tm = time_pat.match(line)
            ts = None
            if tm:
                try:
                    ts = datetime.strptime(tm.group(1), '%Y-%m-%d %H:%M:%S,%f')
                except:
                    pass

            m = pfam_pat.search(line)
            if m:
                vals = [float(x) for x in m.group(1).split(',')]
                pfam_history.append((batch_idx, vals))
                if ts:
                    timestamps.append(ts)
                continue

            m = preg_pat.search(line)
            if m:
                best_f = int(m.group(1))
                num_regions = int(m.group(2))
                vals = [float(x) for x in m.group(3).split(',')][:num_regions]
                preg_history.append((batch_idx, best_f, num_regions, vals))
                region_hist[num_regions] += 1
                continue

            m = batch_pat.search(line)
            if m:
                feedback_counts.append((batch_idx, int(m.group(1))))
                batch_idx += 1
                continue

            m = upd_pat.search(line)
            if m:
                updates.append({
                    'reward': float(m.group(1)),
                    'family': int(m.group(2)),
                    'region': int(m.group(3)),
                    't': int(m.group(4)),
                })
                continue

            m = adapt_pat.search(line)
            if m:
                adapt_events.append((batch_idx, int(m.group(1)), m.group(2)))
                continue

            if split_pat.search(line):
                split_count += 1
                continue

            if merge_pat.search(line):
                merge_count += 1
                continue

    print(f"\n{'='*60}")
    print("  Python Scheduler Log 分析（自适应 region）")
    print(f"  总 batch 数: {batch_idx}")
    print(f"  总 UPDATE 数: {len(updates)}")
    if timestamps and len(timestamps) >= 2:
        dur = (timestamps[-1] - timestamps[0]).total_seconds()
        print(f"  运行时长: {dur:.1f}s ({dur/60:.1f}min)")
        print(f"  平均 batch 速度: {batch_idx/max(dur,0.01):.1f} batch/s")
    print(f"{'='*60}")

    fc = [n for _, n in feedback_counts]
    nonzero = [n for n in fc if n > 0]
    print("\n【1】Feedback 统计")
    print(f"  零反馈 batch: {len(fc)-len(nonzero)} / {len(fc)} ({100*(len(fc)-len(nonzero))/max(len(fc),1):.1f}%)")
    if nonzero:
        print(f"  有反馈 batch: {len(nonzero)} ({100*len(nonzero)/max(len(fc),1):.1f}%)")
        print(f"  反馈数均值: {np.mean(nonzero):.2f}  最大: {max(nonzero)}  总计: {sum(nonzero)}")

    print(f"\n【2】P_fam 演化 (每隔 ~{max(1,len(pfam_history)//10)} 条采样)")
    step = max(1, len(pfam_history) // 10)
    for i in range(0, len(pfam_history), step):
        idx, vals = pfam_history[i]
        print(f"  batch {idx:>5}: [{', '.join(f'{v:.3f}' for v in vals)}]")
    if pfam_history:
        idx, vals = pfam_history[-1]
        print(f"  batch {idx:>5}: [{', '.join(f'{v:.3f}' for v in vals)}]  (最终)")

    print("\n【3】num_regions 与自适应事件")
    for nr in sorted(region_hist):
        cnt = region_hist[nr]
        print(f"  {nr:>2} regions: {cnt:>5} ({100*cnt/max(sum(region_hist.values()),1):.1f}%)")
    print(f"  split 次数: {split_count}")
    print(f"  merge 次数: {merge_count}")
    print(f"  adapt 写回次数: {len(adapt_events)}")
    for batch, nr, bounds in adapt_events[:10]:
        print(f"    batch {batch:>5}: num_regions={nr} bounds={bounds}")
    if len(adapt_events) > 10:
        print(f"    ... 共 {len(adapt_events)} 次")

    print("\n【4】P_reg 熵/区分度")
    step = max(1, len(preg_history) // 10)
    for i in range(0, len(preg_history), step):
        idx, bf, nr, vals = preg_history[i]
        p = np.array(vals, dtype=np.float64)
        spread = float(np.max(p) - np.min(p)) if len(p) else 0.0
        h = entropy(p)
        max_ent = np.log(max(nr, 1)) if nr > 1 else 0.0
        print(f"  batch {idx:>5}: regs={nr:>2} spread={spread:.4f} H={h:.3f}/{max_ent:.3f}")
    if preg_history:
        spreads = np.array([np.max(v) - np.min(v) for _, _, _, v in preg_history])
        print(f"  平均 spread: {spreads.mean():.4f}  中位: {np.median(spreads):.4f}  最大: {spreads.max():.4f}")

    if updates:
        rewards = [u['reward'] for u in updates]
        families = [u['family'] for u in updates]
        regions = [u['region'] for u in updates]
        print(f"\n【5】UPDATE 统计 (共 {len(updates)} 次)")
        print(f"  reward 均值: {np.mean(rewards):.2f}  中位: {np.median(rewards):.2f}  范围: [{min(rewards):.1f}, {max(rewards):.1f}]")
        fc_cnt = Counter(families)
        print("  Family 分布:")
        for f in sorted(fc_cnt, key=lambda x: -fc_cnt[x]):
            print(f"    f={f}: {fc_cnt[f]:>5} ({100*fc_cnt[f]/len(updates):.1f}%)")
        rc_cnt = Counter(regions)
        print("  Region 分布 (Top 8):")
        for r in sorted(rc_cnt, key=lambda x: -rc_cnt[x])[:8]:
            print(f"    r={r:>2}: {rc_cnt[r]:>5} ({100*rc_cnt[r]/len(updates):.1f}%)")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python analyze_py_log.py <py_log文件>")
        sys.exit(1)
    main(sys.argv[1])

