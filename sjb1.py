import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体为Times New Roman并加粗
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'

# 数据准备
frame_rates = np.array([8, 16, 32, 64])

# VILA1.5-long vila 数据
vila_long_ori = np.array([0.6618, 0.657, 0.6425, 0.6425])
vila_long_sub = np.array([0.657, 0.6618, 0.7246, 0.7101])

# Qwen2-VL-7B-Instruct 数据
qwen_ori = np.array([0.5314, 0.5652, 0.6329, 0.6329])
qwen_sub = np.array([0.6039, 0.6473, 0.6618, 0.6522])

# 创建图形和轴
fig, ax = plt.subplots(figsize=(10, 7))

# 使用不同线宽和标记绘制线条
# VILA1.5-long vila ori - 最粗线 + 圆形标记
ax.plot(frame_rates, vila_long_ori, color='black', linewidth=3, marker='o', markersize=8, 
        label='VILA1.5-long vila ori')

# VILA1.5-long vila sub - 粗线 + 方形标记
ax.plot(frame_rates, vila_long_sub, color='black', linewidth=2.5, marker='s', markersize=8, 
        label='VILA1.5-long vila sub')

# Qwen2-VL-7B-Instruct ori - 中等线 + 三角形标记
ax.plot(frame_rates, qwen_ori, color='black', linewidth=2, marker='^', markersize=8, 
        label='Qwen2-VL-7B-Instruct ori')

# Qwen2-VL-7B-Instruct sub - 细线 + 菱形标记
ax.plot(frame_rates, qwen_sub, color='black', linewidth=1.5, marker='D', markersize=8, 
        label='Qwen2-VL-7B-Instruct sub')

# 设置坐标轴标签和标题
ax.set_xlabel('Video Frame Rate', fontsize=20, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=20, fontweight='bold')

# 设置x轴为对数刻度
ax.set_xscale('log')

# 设置x轴刻度
ax.set_xticks([8, 16, 32, 64])
ax.set_xticklabels(['8', '16', '32', '64'], fontsize=16)
ax.tick_params(axis='y', labelsize=16)

# 设置网格
ax.grid(True, alpha=0.3)

# 设置图例
ax.legend(loc='lower right', fontsize=14, frameon=True, fancybox=True, shadow=True)

# 设置坐标轴范围
ax.set_xlim(7, 70)
ax.set_ylim(0.5, 0.8)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('video_frame_rate_impact.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()