import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 数据准备
months = ['3月', '4月', '5月', '6月', '7月']
output_2025 = [0.908, 0.950, 0.634, 0.806, 0.787]
profit_2025 = [1.0, 1.0, 0.0, 0.057, 0.618]
output_2024 = [0.660, 0.999, 0.977, 0.924, 0.870]
profit_2024 = [0.0, 1.044, 0.998, 0.686, 0.490]

# 创建画布
fig, ax1 = plt.subplots(figsize=(12, 6))

# 产值达成率（左轴）
ax1.plot(months, output_2025, 'o-', color='#3498db', linewidth=2, markersize=8, label='2025产值')
ax1.plot(months, output_2024, 'o--', color='#2980b9', linewidth=2, markersize=8, label='2024产值')
ax1.axhline(y=1, color='#e74c3c', linestyle=':', linewidth=2)
ax1.set_ylabel('产值达成率', fontsize=12)
ax1.set_ylim(0, 1.2)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 利润率（右轴）
ax2 = ax1.twinx()
ax2.plot(months, profit_2025, 's-', color='#e67e22', linewidth=2, markersize=8, label='2025利润')
ax2.plot(months, profit_2024, 's--', color='#d35400', linewidth=2, markersize=8, label='2024利润')
ax2.set_ylabel('利润率达成率', fontsize=12)
ax2.set_ylim(-0.2, 1.2)

# 图表装饰
plt.title('产值与利润率达成趋势对比 (2024 vs 2025)', fontsize=14, pad=20)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()

# 数据准备
days_2025 = [91, 75, 77, 77, 79]
days_2024 = [76, 80, 88, 96, 74]

# 创建图表
plt.figure(figsize=(10, 6))
bar_width = 0.35
x = np.arange(len(months))

# 柱状图
bars1 = plt.bar(x - bar_width/2, days_2025, bar_width, color='#9b59b6', label='2025')
bars2 = plt.bar(x + bar_width/2, days_2024, bar_width, color='#3498db', label='2024')

# 目标线
plt.axhline(y=75, color='#e74c3c', linestyle='--', linewidth=2, label='目标值')

# 数据标签
for bar in bars1 + bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.0f}', ha='center', va='bottom')

# 图表装饰
plt.xticks(x, months)
plt.ylabel('周转天数', fontsize=12)
plt.title('应收账款周转天数对比', fontsize=14)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
# 数据准备
cost_ratio = [0.7396, 0.7462, 1.023, 0.8395, 0.8165]
quality_loss = [0.0095, 0.0080, 0.0099, 0.0103, 0.0083]

# 创建画布
fig, ax1 = plt.subplots(figsize=(12, 6))

# 生产成本占比（面积图）
ax1.fill_between(months, cost_ratio, color='#3498db', alpha=0.3, label='生产成本占比')
ax1.plot(months, cost_ratio, 'o-', color='#2980b9', linewidth=2)
ax1.axhline(y=0.8, color='#e74c3c', linestyle='--')
ax1.set_ylabel('生产成本占比', fontsize=12)
ax1.set_ylim(0.6, 1.1)

# 质量损失占比（折线图）
ax2 = ax1.twinx()
ax2.plot(months, quality_loss, 's--', color='#e67e22', linewidth=2, markersize=8, label='质量损失占比')
ax2.axhline(y=0.01, color='#f39c12', linestyle=':')
ax2.set_ylabel('质量损失占比', fontsize=12)
ax2.set_ylim(0.005, 0.015)

# 图表装饰
plt.title('生产成本与质量损失趋势分析', fontsize=14)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
from math import pi

# 数据准备
categories = ['生产一次合格率', '出缸合格率', '成品不良率', '退货率']
targets = [0.9, 0.94, 0.011, 0.005]
values = [0.745, 0.9402, 0.0085, 0.0057]

# 角度计算
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# 雷达图设置
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# 绘制目标线
targets += targets[:1]  # 修复：正确闭合目标值数据
ax.plot(angles, targets, '--', color='#e74c3c', linewidth=2, label='目标值')
ax.fill(angles, targets, color='#e74c3c', alpha=0.1)

# 绘制实际值
values += values[:1]
ax.plot(angles, values, 'o-', color='#3498db', linewidth=2, label='实际值')
ax.fill(angles, values, color='#3498db', alpha=0.25)

# 坐标轴设置
plt.xticks(angles[:-1], categories, fontsize=12)
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2","0.4","0.6","0.8","1.0"], color="grey", size=10)
plt.ylim(0, 1.1)

# 添加图例和标题
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.title('2025年部门关键质量指标雷达图', fontsize=14, pad=20)
plt.tight_layout()
plt.show()
# 数据准备
energy_data = {
    '月份': months,
    '用水量': [108.6, 108.24, 142.34, 127.64, 106.96],
    '用电量': [871.77, 963.98, 992.46, 1001.57, 1005.53],
    '用气量': [4.51, 4.57, 5.06, 4.31, 3.53]
}

# 创建图表
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# 用水量
axes[0].bar(months, energy_data['用水量'], color='#3498db')
axes[0].axhline(y=100, color='r', linestyle='--')
axes[0].set_title('吨布用水量 (目标≤100)', fontsize=12)
axes[0].grid(axis='y', linestyle='--')

# 用电量
axes[1].bar(months, energy_data['用电量'], color='#2ecc71')
axes[1].axhline(y=1000, color='r', linestyle='--')
axes[1].set_title('吨布用电量 (目标≤1000度)', fontsize=12)
axes[1].grid(axis='y', linestyle='--')

# 用气量
axes[2].bar(months, energy_data['用气量'], color='#e67e22')
axes[2].axhline(y=4.2, color='r', linestyle='--')
axes[2].set_title('吨布用气量 (目标≤4.2吨)', fontsize=12)
axes[2].grid(axis='y', linestyle='--')

plt.suptitle('2025年能源消耗指标监控', fontsize=14, y=0.95)
plt.tight_layout()
plt.show()
# 数据准备
safety_data = {
    '月份': months,
    '安全事件': [0, 0, 1, 0, 1],
    '环保达标': [0, 1, 1, 0, 0]
}

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 安全事件饼图
ax1.pie([sum(safety_data['安全事件']), 5-sum(safety_data['安全事件'])],
        labels=['发生', '未发生'],
        colors=['#e74c3c', '#2ecc71'],
        autopct='%1.1f%%',
        startangle=90)
ax1.set_title('2025年安全事件统计(1-7月)', fontsize=12)

# 环保达标率饼图
ax2.pie([sum(safety_data['环保达标']), 3],
        labels=['达标', '未达标'],
        colors=['#2ecc71', '#f39c12'],
        autopct='%1.1f%%',
        startangle=90)
ax2.set_title('环保合规达标率(有效月份)', fontsize=12)

plt.suptitle('安全环保绩效分析', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()