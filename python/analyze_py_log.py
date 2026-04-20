import sys
import re
import numpy as np
from collections import defaultdict
from datetime import datetime

def parse_log(filepath):
    records = []
    current = {}

    preg_pat  = re.compile(r'\[PY\]\[SEND\] P_reg=\[(.*?)\]')
    pfam_pat  = re.compile(r'\[PY\]\[SEND\] best_r=(\d+) P_fam\[\d+\]=\[(.*?)\]')
    done_pat  = re.compile(r'\[PY\]\[BATCH_DONE\] reward=([\d.]+).*?best_r=(-?\d+).*?best_f=(-?\d+)')
    upd_pat   = re.compile(r'\[PY\]\[UPDATE\] reward=([\d.]+) baseline=([\d.]+) advantage=(-?[\d.]+) best_r=(\d+) best_f=(\d+)')
    time_pat  = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)')

    def flush():
        if current:
            records.append(dict(current))
            current.clear()

    with open(filepath) as f:
        for line in f:
            tm = time_pat.match(line)
            timestamp = None
            if tm:
                try:
                    timestamp = datetime.strptime(tm.group(1), '%Y-%m-%d %H:%M:%S,%f')
                except:
                    pass

            m = preg_pat.search(line)
            if m:
                flush()
                current['P_reg'] = [float(x) for x in m.group(1).split(',')]
                current['timestamp'] = timestamp
                continue

            m = pfam_pat.search(line)
            if m:
                current['best_r_send'] = int(m.group(1))
                current['P_fam_best'] = [float(x) for x in m.group(2).split(',')]
                continue

            m = done_pat.search(line)
            if m:
                current['reward'] = float(m.group(1))
                current['best_r'] = int(m.group(2))
                current['best_f'] = int(m.group(3))
                continue

            m = upd_pat.search(line)
            if m:
                current['baseline'] = float(m.group(2))
                current['advantage'] = float(m.group(3))
                continue

    flush()
    return records


def analyze(records):
    print(f"\n{'='*60}")
    print(f"  共解析 {len(records)} 条记录")
    print(f"{'='*60}")

    # ── 1. reward 分布 ──
    rewards = [r['reward'] for r in records if 'reward' in r]
    pos = [r for r in rewards if r > 0]
    print(f"\n【1】Reward 统计")
    print(f"  总记录数: {len(rewards)}")
    print(f"  有正收益: {len(pos)} / {len(rewards)}  ({100*len(pos)/max(len(rewards),1):.1f}%)")
    if pos:
        print(f"  正收益均值: {np.mean(pos):.2f}  最大: {max(pos):.2f}  中位: {np.median(pos):.2f}")

    # ── 2. advantage 分布 ──
    advs = [r['advantage'] for r in records if 'advantage' in r]
    if advs:
        pos_adv = sum(1 for a in advs if a > 0)
        print(f"\n【2】Advantage 分布（共 {len(advs)} 次更新）")
        print(f"  正advantage: {pos_adv}  负advantage: {len(advs)-pos_adv}")
        print(f"  均值: {np.mean(advs):.3f}  std: {np.std(advs):.3f}")
        print(f"  范围: [{min(advs):.2f}, {max(advs):.2f}]")

    # ── 3. P_reg 各维度均值与方差 ──
    pregs = [r['P_reg'] for r in records if 'P_reg' in r and len(r['P_reg']) == 16]
    if pregs:
        arr = np.array(pregs)
        means = arr.mean(axis=0)
        stds  = arr.std(axis=0)
        print(f"\n【3】P_reg 各region平均概率（共 {len(pregs)} 条）")
        print(f"  {'Region':>8} {'均值':>8} {'std':>8} {'评价':>6}")
        overall_mean = means.mean()
        for i in range(16):
            flag = '★' if means[i] > overall_mean * 1.3 else ('↓' if means[i] < overall_mean * 0.7 else ' ')
            print(f"  {i:>8} {means[i]:>8.4f} {stds[i]:>8.4f} {flag:>6}")

    # ── 4. P_fam 各family平均概率 ──
    pfams = [r['P_fam_best'] for r in records if 'P_fam_best' in r and len(r.get('P_fam_best', [])) == 5]
    if pfams:
        arr = np.array(pfams)
        means = arr.mean(axis=0)
        print(f"\n【4】P_fam 各family平均概率（基于best_r视角，共 {len(pfams)} 条）")
        fam_names = ['f0(bit/byte flip)', 'f1(arith)', 'f2(insert)', 'f3(splice)', 'f4(special)']
        for i, name in enumerate(fam_names):
            bar = '█' * int(means[i] * 40)
            print(f"  {name:<20} {means[i]:.3f}  {bar}")

    # ── 5. best_r 分布（C侧反馈的最佳region）──
    best_rs = [r['best_r'] for r in records if 'best_r' in r and r['best_r'] >= 0]
    if best_rs:
        from collections import Counter
        cnt = Counter(best_rs)
        print(f"\n【5】C侧反馈的best_r分布（共 {len(best_rs)} 次正收益）")
        for r_id in sorted(cnt, key=lambda x: -cnt[x])[:8]:
            bar = '█' * cnt[r_id]
            print(f"  region {r_id:>2}: {cnt[r_id]:>4}次  {bar}")

    # ── 6. best_f 分布（C侧反馈的最佳family）──
    best_fs = [r['best_f'] for r in records if 'best_f' in r and r['best_f'] >= 0]
    if best_fs:
        from collections import Counter
        cnt = Counter(best_fs)
        print(f"\n【6】C侧反馈的best_f分布（共 {len(best_fs)} 次正收益）")
        for f_id in sorted(cnt, key=lambda x: -cnt[x]):
            bar = '█' * cnt[f_id]
            print(f"  family {f_id}: {cnt[f_id]:>4}次  {bar}")

    # ── 7. baseline 收敛趋势 ──
    baselines = [(i, r['baseline']) for i, r in enumerate(records) if 'baseline' in r]
    if baselines:
        print(f"\n【7】Baseline 变化趋势（每隔若干次采样）")
        step = max(1, len(baselines) // 10)
        for idx, bl in baselines[::step]:
            bar = '█' * int(bl / max(b for _, b in baselines) * 30) if max(b for _, b in baselines) > 0 else ''
            print(f"  update #{idx:>5}: baseline={bl:.3f}  {bar}")

    # ── 8. 时间间隔分析（batch处理速度）──
    times = [r['timestamp'] for r in records if r.get('timestamp')]
    if len(times) > 2:
        diffs = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
        diffs = [d for d in diffs if 0 < d < 300]  # 过滤掉超长等待
        if diffs:
            print(f"\n【8】Batch 处理速度")
            print(f"  平均间隔: {np.mean(diffs):.3f}s")
            print(f"  最快: {min(diffs):.3f}s  最慢: {max(diffs):.3f}s")
            print(f"  估计吞吐: {1/np.mean(diffs):.1f} batch/s")

    # ── 9. P_reg 熵（多样性）趋势 ──
    if pregs:
        entropies = []
        for p in pregs:
            p = np.array(p)
            p = p[p > 1e-12]
            entropies.append(-np.sum(p * np.log(p)))
        max_entropy = np.log(16)
        print(f"\n【9】P_reg 分布熵（多样性）趋势")
        print(f"  最大可能熵（均匀分布）: {max_entropy:.3f}")
        step = max(1, len(entropies) // 8)
        for i in range(0, len(entropies), step):
            h = entropies[i]
            bar = '█' * int(h / max_entropy * 30)
            print(f"  record #{i:>5}: H={h:.3f}  {bar}")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python analyze_py_log.py <py_log文件>")
        sys.exit(1)
    records = parse_log(sys.argv[1])
    analyze(records)