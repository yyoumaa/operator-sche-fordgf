"""
分析 /two-stage/afl_to_python.txt (变异结果日志)

格式: --newcase-- 分隔每个种子
每行: reward op1 pos1 op2 pos2 ...  (第一个数是 reward, 后面是 op pos 对)
空 case (只有 --newcase--) 表示该种子没有产生有效变异

提取:
  - 每个 case 的 reward, 算子数, 算子分布
  - 总体 reward 分布
  - 算子使用频率
  - 早期 vs 晚期对比 (随机 havoc vs bandit 控制)
  - 单算子 case 比例 (bandit 接管后的特征)
"""
import sys
import numpy as np
from collections import Counter, defaultdict

def main(filepath):
    cases = []
    current_lines = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line == '--newcase--':
                if current_lines:
                    cases.append(current_lines)
                    current_lines = []
                else:
                    cases.append([])
                continue
            if line:
                current_lines.append(line)
        if current_lines:
            cases.append(current_lines)

    total_cases = len(cases)
    empty_cases = sum(1 for c in cases if len(c) == 0)

    all_rewards = []
    all_ops = Counter()
    ops_per_case = []
    single_op_cases = 0
    single_op_dist = Counter()

    case_details = []

    for c in cases:
        if not c:
            case_details.append(None)
            continue
        for line in c:
            nums = list(map(int, line.split()))
            if len(nums) < 1:
                continue
            reward = nums[0]
            all_rewards.append(reward)
            ops_in_line = []
            i = 1
            while i + 1 < len(nums):
                op = nums[i]
                pos = nums[i+1]
                ops_in_line.append((op, pos))
                all_ops[op] += 1
                i += 2
            ops_per_case.append(len(ops_in_line))
            if len(ops_in_line) == 1:
                single_op_cases += 1
                single_op_dist[ops_in_line[0][0]] += 1
            case_details.append({
                'reward': reward,
                'num_ops': len(ops_in_line),
                'ops': ops_in_line,
            })

    print(f"\n{'='*60}")
    print(f"  Mutation Log 分析")
    print(f"  总 case 数: {total_cases}  空 case: {empty_cases}")
    print(f"  有效变异行数: {len(all_rewards)}")
    print(f"{'='*60}")

    # ── 1. Reward 分布 ──
    if all_rewards:
        arr = np.array(all_rewards)
        pos = arr[arr > 0]
        print(f"\n【1】Reward 分布")
        print(f"  正: {len(pos)} ({100*len(pos)/len(arr):.1f}%)")
        print(f"  零: {np.sum(arr==0)} ({100*np.sum(arr==0)/len(arr):.1f}%)")
        print(f"  负: {np.sum(arr<0)}")
        if len(pos) > 0:
            print(f"  正 reward: 均值={pos.mean():.1f} 中位={np.median(pos):.0f} "
                  f"范围=[{pos.min()}, {pos.max()}]")
        buckets = Counter()
        for r in pos:
            b = int(r // 10) * 10
            buckets[b] += 1
        if buckets:
            print(f"  正 reward 分桶:")
            for k in sorted(buckets):
                bar = '█' * min(buckets[k], 50)
                print(f"    [{k:>3}-{k+10:>3}): {buckets[k]:>5}  {bar}")

    # ── 2. 算子使用频率 ──
    print(f"\n【2】算子使用频率 (Top 15)")
    total_op_uses = sum(all_ops.values())
    for op, cnt in sorted(all_ops.items(), key=lambda x: -x[1])[:15]:
        bar = '█' * (cnt * 40 // max(total_op_uses, 1))
        print(f"  op={op:>2}: {cnt:>7} ({100*cnt/total_op_uses:.1f}%)  {bar}")

    # ── 3. 每行算子数分布 ──
    if ops_per_case:
        arr = np.array(ops_per_case)
        print(f"\n【3】每行算子数分布")
        print(f"  均值: {arr.mean():.1f}  中位: {np.median(arr):.0f}  "
              f"范围: [{arr.min()}, {arr.max()}]")
        print(f"  单算子行: {single_op_cases} ({100*single_op_cases/len(arr):.1f}%)")
        print(f"  多算子行 (>=10): {np.sum(arr>=10)} ({100*np.sum(arr>=10)/len(arr):.1f}%)")

    # ── 4. 单算子 case 的算子分布 (bandit 控制特征) ──
    if single_op_dist:
        print(f"\n【4】单算子 case 的算子分布 (bandit 控制)")
        for op, cnt in sorted(single_op_dist.items(), key=lambda x: -x[1]):
            bar = '█' * (cnt * 30 // max(single_op_dist.values()))
            print(f"  op={op:>2}: {cnt:>5}  {bar}")

    # ── 5. 早期 vs 晚期对比 ──
    n = len(all_rewards)
    if n > 100:
        early = all_rewards[:n//5]
        late = all_rewards[-n//5:]
        e_pos = sum(1 for r in early if r > 0)
        l_pos = sum(1 for r in late if r > 0)
        e_ops = ops_per_case[:n//5] if len(ops_per_case) >= n//5 else ops_per_case[:len(ops_per_case)//5]
        l_ops = ops_per_case[-n//5:] if len(ops_per_case) >= n//5 else ops_per_case[-len(ops_per_case)//5:]
        print(f"\n【5】早期 vs 晚期对比 (前20% vs 后20%)")
        print(f"  早期: {len(early)}行, 正reward {e_pos} ({100*e_pos/max(len(early),1):.1f}%), "
              f"平均算子数 {np.mean(e_ops):.1f}")
        print(f"  晚期: {len(late)}行, 正reward {l_pos} ({100*l_pos/max(len(late),1):.1f}%), "
              f"平均算子数 {np.mean(l_ops):.1f}")

    # ── 6. Reward 趋势 (滑动窗口) ──
    if len(all_rewards) > 50:
        print(f"\n【6】Reward 趋势 (滑动窗口)")
        window = max(1, len(all_rewards) // 20)
        for start in range(0, len(all_rewards), window):
            chunk = all_rewards[start:start+window]
            pos_cnt = sum(1 for r in chunk if r > 0)
            avg_r = np.mean([r for r in chunk if r > 0]) if pos_cnt > 0 else 0
            bar = '█' * pos_cnt + '░' * (len(chunk) - pos_cnt)
            print(f"  [{start:>5}-{start+len(chunk):>5}] {bar}  "
                  f"正:{pos_cnt}/{len(chunk)} avg_pos={avg_r:.1f}")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python analyze_mutation.py <afl_to_python.txt>")
        sys.exit(1)
    main(sys.argv[1])
