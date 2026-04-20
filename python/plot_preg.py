import sys
import re
import matplotlib.pyplot as plt

def parse_preg(filepath):
    pattern = re.compile(r'\[PY\]\[SEND\] P_reg=\[(.*?)\]')
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                values = [float(x.strip()) for x in m.group(1).split(',')]
                records.append(values)
    return records

def main():
    if len(sys.argv) < 2:
        print("用法: python plot_preg.py <log文件路径>")
        sys.exit(1)

    filepath = sys.argv[1]
    records = parse_preg(filepath)
    total = len(records)
    print(f"共找到 {total} 条 P_reg 记录")

    user_input = input("请输入需要绘图的范围（如 200-3000）: ").strip()
    try:
        start, end = [int(x) for x in user_input.split('-')]
    except:
        print("输入格式错误，请使用如 200-3000 的格式")
        sys.exit(1)

    start = max(0, start)
    end = min(total, end)
    selected = records[start:end]
    n = len(selected)
    if n == 0:
        print("选定范围内没有数据")
        sys.exit(1)

    print(f"绘制第 {start} 到 {end} 条，共 {n} 条记录")

    # 转置：每个维度一条线
    num_dims = len(selected[0])
    xs = list(range(start, start + n))
    dims = [[selected[i][d] for i in range(n)] for d in range(num_dims)]

    plt.figure(figsize=(16, 6))
    colors = plt.cm.tab20.colors
    for d in range(num_dims):
        plt.plot(xs, dims[d], label=f'region {d}', color=colors[d % len(colors)], linewidth=0.8)

    plt.xlabel('记录编号')
    plt.ylabel('概率值')
    plt.title(f'P_reg 各维度变化趋势（第 {start}~{end} 条）')
    plt.legend(loc='upper right', ncol=4, fontsize=7)
    plt.tight_layout()
    plt.savefig(f'{sys.argv[2]}', dpi=150)
    print("图已保存为 preg_plot.png")
    plt.show()

if __name__ == '__main__':
    main()