import re
import matplotlib.pyplot as plt
from collections import Counter

# ====== 1. 读取文件 ======
file_path = "../mutate_log.txt"  # 替换成你的文件路径

pattern = re.compile(r"\[C Mock\].*变异区域:\s*(\d+),\s*算子族:\s*(\d+)")

mutation_regions = []
operator_families = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            region = int(match.group(1))
            operator = int(match.group(2))

            mutation_regions.append(region)
            operator_families.append(operator)

print(f"total {len(mutation_regions)} ")

# ====== 2. 统计分布 ======
region_counts = Counter(mutation_regions)
operator_counts = Counter(operator_families)

# ====== 3. 画饼图 ======
plt.figure(figsize=(12, 5))

# 变异区域饼图
plt.subplot(1, 2, 1)
plt.pie(
    region_counts.values(),
    labels=region_counts.keys(),
    autopct='%1.1f%%'
)
plt.title("Region")

# 算子族饼图
plt.subplot(1, 2, 2)
plt.pie(
    operator_counts.values(),
    labels=operator_counts.keys(),
    autopct='%1.1f%%'
)
plt.title("family")

plt.tight_layout()
plt.savefig("pie_chart.png", dpi=300)  # 保存
plt.show()

# ====== 4. 折线图（按顺序） ======
x = list(range(1, len(mutation_regions) + 1))

plt.figure(figsize=(12, 8))

# 算子族折线图
plt.subplot(2, 1, 1)
plt.plot(x, operator_families, marker='o')
plt.title("time")
plt.xlabel("times")
plt.ylabel("family")

# 变异区域折线图
plt.subplot(2, 1, 2)
plt.plot(x, mutation_regions, marker='o')
plt.title("region")
plt.xlabel("times")
plt.ylabel("region")

plt.tight_layout()
plt.savefig("trend_chart.png", dpi=300)  # 保存
plt.show()