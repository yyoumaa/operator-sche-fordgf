import sys
import re
import json
from collections import defaultdict

def parse_log(filepath):
    seeds = []
    current_seed = None
    current_trials = []

    seed_pat    = re.compile(r'\[BANDIT\]\[SEED\] len=(\d+).*?prox=(\d+).*?ema=([\d.]+).*?dynmax=(\d+)')
    feat_pat    = re.compile(r'\[BANDIT\]\[REGION_FEAT\] i=(\d+).*?hrew=([\d.]+).*?hcov=([\d.]+)')
    try_pat     = re.compile(r'\[BANDIT\]\[TRY\] trial=(\d+) sampled_r=(\d+) sampled_f=(\d+) op=(\d+) pos=(\d+) len=(\d+) in_region=(\d+) reward=(-?\d+)')
    batch_pat   = re.compile(r'\[BANDIT\]\[BATCH_END\] best_r=(-?\d+) best_f=(-?\d+) batch_reward=([\d.]+)')
    sample_pat  = re.compile(r'\[BANDIT\]\[SAMPLE\] trial=(\d+) sampled_r=(\d+) sampled_f=(\d+)')

    with open(filepath, 'r') as f:
        for line in f:
            m = seed_pat.search(line)
            if m:
                if current_seed is not None:
                    current_seed['trials'] = current_trials
                    seeds.append(current_seed)
                current_seed = {
                    'len': int(m.group(1)),
                    'prox': int(m.group(2)),
                    'ema': float(m.group(3)),
                    'dynmax': int(m.group(4)),
                    'hrew': [0.0] * 16,
                    'hcov': [0.0] * 16,
                    'trials': [],
                    'best_r': -1,
                    'best_f': -1,
                    'batch_reward': 0.0,
                }
                current_trials = []
                continue

            if current_seed is None:
                continue

            m = feat_pat.search(line)
            if m:
                i = int(m.group(1))
                current_seed['hrew'][i] = float(m.group(2))
                current_seed['hcov'][i] = float(m.group(3))
                continue

            m = try_pat.search(line)
            if m:
                current_trials.append({
                    'trial': int(m.group(1)),
                    'sampled_r': int(m.group(2)),
                    'sampled_f': int(m.group(3)),
                    'op': int(m.group(4)),
                    'pos': int(m.group(5)),
                    'len': int(m.group(6)),
                    'in_region': int(m.group(7)),
                    'reward': int(m.group(8)),
                })
                continue

            m = batch_pat.search(line)
            if m:
                current_seed['best_r'] = int(m.group(1))
                current_seed['best_f'] = int(m.group(2))
                current_seed['batch_reward'] = float(m.group(3))
                continue

    if current_seed is not None:
        current_seed['trials'] = current_trials
        seeds.append(current_seed)

    return seeds


def analyze(seeds):
    print(f"\n{'='*60}")
    print(f"  共解析 {len(seeds)} 个seed批次")
    print(f"{'='*60}")

    # ── 1. 总体reward分布 ──
    rewards = [s['batch_reward'] for s in seeds]
    positive = [r for r in rewards if r > 0]
    print(f"\n【1】Batch Reward 统计")
    print(f"  有正收益批次: {len(positive)} / {len(seeds)}  ({100*len(positive)/max(len(seeds),1):.1f}%)")
    if positive:
        print(f"  正收益均值: {sum(positive)/len(positive):.2f}")
        print(f"  最大reward: {max(positive):.2f}")
        print(f"  reward分布: ", end="")
        buckets = defaultdict(int)
        for r in positive:
            b = int(r // 20) * 20
            buckets[b] += 1
        for k in sorted(buckets):
            print(f"[{k}-{k+20}): {buckets[k]}", end="  ")
        print()

    # ── 2. in_region命中率 ──
    total_trials = 0
    hit_trials = 0
    for s in seeds:
        for t in s['trials']:
            total_trials += 1
            if t['in_region'] == 1:
                hit_trials += 1
    print(f"\n【2】采样命中率（变异实际落在采样region内）")
    print(f"  总trial数: {total_trials}")
    print(f"  in_region=1: {hit_trials}  ({100*hit_trials/max(total_trials,1):.1f}%)")

    # ── 3. 各region被采样次数 vs 产生正reward次数 ──
    region_sampled = defaultdict(int)
    region_pos_reward = defaultdict(int)
    region_total_reward = defaultdict(float)
    for s in seeds:
        for t in s['trials']:
            r = t['sampled_r']
            region_sampled[r] += 1
            if t['reward'] > 0:
                region_pos_reward[r] += 1
                region_total_reward[r] += t['reward']

    print(f"\n【3】各Region采样频次 vs 正reward次数")
    print(f"  {'Region':>8} {'采样次数':>10} {'正reward次':>12} {'命中率%':>10} {'累计reward':>12}")
    for r in range(16):
        n = region_sampled[r]
        p = region_pos_reward[r]
        rate = 100*p/n if n > 0 else 0
        total_r = region_total_reward[r]
        print(f"  {r:>8} {n:>10} {p:>12} {rate:>10.1f} {total_r:>12.1f}")

    # ── 4. 各family被采样次数 vs 正reward次数 ──
    fam_sampled = defaultdict(int)
    fam_pos_reward = defaultdict(int)
    fam_total_reward = defaultdict(float)
    for s in seeds:
        for t in s['trials']:
            f = t['sampled_f']
            fam_sampled[f] += 1
            if t['reward'] > 0:
                fam_pos_reward[f] += 1
                fam_total_reward[f] += t['reward']

    print(f"\n【4】各Family采样频次 vs 正reward次数")
    print(f"  {'Family':>8} {'采样次数':>10} {'正reward次':>12} {'命中率%':>10} {'累计reward':>12}")
    for f in range(5):
        n = fam_sampled[f]
        p = fam_pos_reward[f]
        rate = 100*p/n if n > 0 else 0
        total_r = fam_total_reward[f]
        print(f"  {f:>8} {n:>10} {p:>12} {rate:>10.1f} {total_r:>12.1f}")

    # ── 5. 各算子正reward统计 ──
    op_pos = defaultdict(int)
    op_total = defaultdict(float)
    op_count = defaultdict(int)
    for s in seeds:
        for t in s['trials']:
            op = t['op']
            op_count[op] += 1
            if t['reward'] > 0:
                op_pos[op] += 1
                op_total[op] += t['reward']

    print(f"\n【5】各算子正reward统计（只显示有正收益的）")
    print(f"  {'Op':>5} {'执行次数':>10} {'正reward次':>12} {'累计reward':>12}")
    for op in sorted(op_total, key=lambda x: -op_total[x]):
        print(f"  {op:>5} {op_count[op]:>10} {op_pos[op]:>12} {op_total[op]:>12.1f}")

    # ── 6. hrew分布随时间变化趋势 ──
    print(f"\n【6】hrew特征积累情况（每50个seed采样一次均值）")
    step = max(1, len(seeds) // 10)
    print(f"  {'seed#':>7}", end="")
    for i in range(16):
        print(f"  r{i:02d}", end="")
    print()
    for idx in range(0, len(seeds), step):
        s = seeds[idx]
        print(f"  {idx:>7}", end="")
        for i in range(16):
            val = s['hrew'][i]
            print(f"  {val:.3f}", end="")
        print()

    # ── 7. 零reward比例随时间变化（学习是否在改善）──
    print(f"\n【7】每50个batch的正reward比例变化趋势")
    window = max(1, len(seeds) // 20)
    for start in range(0, len(seeds), window):
        chunk = seeds[start:start+window]
        pos = sum(1 for s in chunk if s['batch_reward'] > 0)
        bar = '█' * pos + '░' * (len(chunk) - pos)
        print(f"  [{start:4d}-{start+len(chunk):4d}] {bar}  {pos}/{len(chunk)}")

    # ── 8. 负reward来源分析 ──
    neg_by_op = defaultdict(int)
    neg_by_region = defaultdict(int)
    for s in seeds:
        for t in s['trials']:
            if t['reward'] < 0:
                neg_by_op[t['op']] += 1
                neg_by_region[t['sampled_r']] += 1

    print(f"\n【8】负reward来源（Top5算子 / Top5 Region）")
    print("  Top5算子:")
    for op, cnt in sorted(neg_by_op.items(), key=lambda x: -x[1])[:5]:
        print(f"    op={op}: {cnt}次")
    print("  Top5 Region:")
    for r, cnt in sorted(neg_by_region.items(), key=lambda x: -x[1])[:5]:
        print(f"    region={r}: {cnt}次")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python analyze_bandit.py <bandit_log文件>")
        sys.exit(1)
    seeds = parse_log(sys.argv[1])
    analyze(seeds)