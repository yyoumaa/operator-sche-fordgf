"""
分析 /operator-sche-fordgf/bandit_log.txt (C 侧 bandit 日志)
逐行流式解析，不全量加载 trials 到内存。

适配自适应 region 新格式：
  - [BANDIT][SEED] len=... num_regions=...
  - [BANDIT][REGION_FEAT] i=... start=... end=...

提取:
  - SEED 特征演化 (prox, ema, global features, num_regions)
  - REGION_FEAT hrew/hcov 积累趋势
  - region 数变化
"""
import sys, re
import numpy as np
from collections import Counter


def entropy(vals):
    arr = np.array(vals, dtype=np.float64)
    arr = arr[arr > 1e-12]
    if len(arr) == 0:
        return 0.0
    return float(-np.sum(arr * np.log(arr)))


def main(filepath):
    seed_pat = re.compile(
        r'\[BANDIT\]\[SEED\] len=(\d+) num_regions=(\d+) prox=(\d+) '
        r'g0=([\d.]+) g1=([\d.]+) g2=([\d.]+) ema=([\d.]+) dynmax=(\d+)')
    feat_pat = re.compile(
        r'\[BANDIT\]\[REGION_FEAT\] i=(\d+) start=(\d+) end=(\d+) '
        r'ent=([\d.]+) pr=([\d.]+) hrew=([\d.]+) hcov=([\d.]+)')

    num_seeds = 0
    seed_snapshots = []
    cur_hrew = []
    current_num_regions = None
    hrew_spread = []
    num_region_hist = Counter()

    with open(filepath) as f:
        for line in f:
            m = seed_pat.search(line)
            if m:
                if cur_hrew and current_num_regions is not None:
                    vals = cur_hrew[:current_num_regions]
                    if vals:
                        hrew_spread.append({
                            'num_regions': current_num_regions,
                            'max': max(vals),
                            'min': min(vals),
                            'spread': max(vals) - min(vals),
                            'entropy': entropy(np.array(vals) / max(np.sum(vals), 1e-12)),
                        })
                num_seeds += 1
                current_num_regions = int(m.group(2))
                cur_hrew = [0.5] * current_num_regions
                num_region_hist[current_num_regions] += 1
                seed_snapshots.append({
                    'len': int(m.group(1)),
                    'num_regions': current_num_regions,
                    'prox': int(m.group(3)),
                    'g0': float(m.group(4)),
                    'g1': float(m.group(5)),
                    'g2': float(m.group(6)),
                    'ema': float(m.group(7)),
                    'dynmax': int(m.group(8)),
                })
                continue

            m = feat_pat.search(line)
            if m and current_num_regions is not None:
                i = int(m.group(1))
                if 0 <= i < current_num_regions:
                    cur_hrew[i] = float(m.group(6))
                continue

    if cur_hrew and current_num_regions is not None:
        vals = cur_hrew[:current_num_regions]
        if vals:
            hrew_spread.append({
                'num_regions': current_num_regions,
                'max': max(vals),
                'min': min(vals),
                'spread': max(vals) - min(vals),
                'entropy': entropy(np.array(vals) / max(np.sum(vals), 1e-12)),
            })

    print(f"\n{'='*60}")
    print("  C 侧 Bandit Log 分析（自适应 region）")
    print(f"  总 seed 数: {num_seeds}")
    print(f"{'='*60}")

    if seed_snapshots:
        print("\n【1】num_regions 分布")
        for nr in sorted(num_region_hist):
            cnt = num_region_hist[nr]
            print(f"  {nr:>2} regions: {cnt:>5} ({100*cnt/len(seed_snapshots):.1f}%)")

        print("\n【2】SEED 特征演化 (采样)")
        step = max(1, len(seed_snapshots) // 8)
        print(f"  {'seed#':>7} {'prox':>6} {'ema':>8} {'g0':>8} {'g1':>6} {'g2':>8} {'len':>6} {'regs':>5}")
        for i in range(0, len(seed_snapshots), step):
            s = seed_snapshots[i]
            print(f"  {i:>7} {s['prox']:>6} {s['ema']:>8.4f} {s['g0']:>8.4f} {s['g1']:>6.2f} {s['g2']:>8.5f} {s['len']:>6} {s['num_regions']:>5}")
        s = seed_snapshots[-1]
        print(f"  {len(seed_snapshots)-1:>7} {s['prox']:>6} {s['ema']:>8.4f} {s['g0']:>8.4f} {s['g1']:>6.2f} {s['g2']:>8.5f} {s['len']:>6} {s['num_regions']:>5}  (最终)")

    if hrew_spread:
        print("\n【3】Region hrew 区分度 (采样)")
        step = max(1, len(hrew_spread) // 8)
        print(f"  {'seed#':>7} {'regs':>5} {'spread':>8} {'max':>8} {'min':>8} {'H(p)':>8}")
        for i in range(0, len(hrew_spread), step):
            s = hrew_spread[i]
            print(f"  {i:>7} {s['num_regions']:>5} {s['spread']:>8.4f} {s['max']:>8.4f} {s['min']:>8.4f} {s['entropy']:>8.4f}")
        arr = np.array([x['spread'] for x in hrew_spread])
        print(f"\n  平均 spread: {arr.mean():.4f}  中位: {np.median(arr):.4f}  最大: {arr.max():.4f}")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python analyze_bandit.py <bandit_log文件>")
        sys.exit(1)
    main(sys.argv[1])

