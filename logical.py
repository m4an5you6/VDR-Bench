# import matplotlib.pyplot as plt
# import numpy as np

# # 设置全局字体为Times New Roman并加粗
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.weight'] = 'bold'

# # Data setup
# categories = ['No Fine-Tuning', 'Supervised', 'QLoRA 4-bit', 'QLoRA 8-bit', 'After VDAT']

# # Original data (ori_correct, sub_correct)
# qwen25_data = [(0.6226, 0.5155), (0.6407, 0.6119), (0.7028, 0.6223), (0.7028, 0.6234), (0.5729, 0.5476), (0.6667, 0.6705), (0.7667, 0.6667)]
# qwen2_data = [(0.5990, 0.6797), (0.6594, 0.7627), (0.6475, 0.7095), (0.6418, 0.7141), (0.6633, 0.7451), (0.7691, 0.7417), (0.8116, 0.8023)]
# video_llama_data = [(0.7752, 0.7400), (0.7349, 0.7812), (None, None), (None, None), (0.7810, 0.7521), (0.8043, 0.7801), (None, None)]

# # 动态计算beta值的函数
# def calculate_adaptive_beta(ori_correct, base_beta=0.5, threshold=0.7):
#     """
#     根据主问题正确率动态计算beta值
#     - 当ori_correct > threshold时，beta偏向0（更关注相对性能）
#     - 当ori_correct < threshold时，beta偏向1（更关注绝对性能）
#     """
#     if ori_correct is None:
#         return base_beta
    
#     # 使用sigmoid函数来平滑过渡
#     # 当ori_correct增加时，beta减小，更关注相对性能
#     adaptive_factor = 1 / (1 + np.exp(5 * (ori_correct - threshold)))
#     beta = base_beta + 0.3 * (0.5 - adaptive_factor)
    
#     # 限制beta在合理范围内
#     return np.clip(beta, 0.2, 0.8)

# # Calculate HL values using adaptive beta
# def calculate_hl(ori_correct, sub_correct, beta=None):
#     if ori_correct is None or ori_correct == 0:
#         return np.nan
    
#     # 如果没有提供beta，则使用自适应计算
#     if beta is None:
#         beta = calculate_adaptive_beta(ori_correct)
    
#     # HL = S_i * A_i^(beta-1)
#     return sub_correct * (ori_correct ** (beta - 1))

# # Calculate values for each model with adaptive beta
# qwen25_values = []
# qwen2_values = []
# video_llama_values = []

# qwen25_betas = []
# qwen2_betas = []
# video_llama_betas = []

# # Map to display names: No Fine-Tuning, Supervised, QLoRA 4-bit Full, QLoRA 8-bit Full, After VDAT
# # We only display 5 categories in the chart, so we select appropriate data points
# display_indices = [0, 1, 4, 5, 6]

# for i in display_indices:
#     if qwen25_data[i][0] is not None:
#         beta = calculate_adaptive_beta(qwen25_data[i][0])
#         qwen25_betas.append(beta)
#         qwen25_values.append(calculate_hl(qwen25_data[i][0], qwen25_data[i][1], beta))
#     else:
#         qwen25_betas.append(np.nan)
#         qwen25_values.append(np.nan)
        
#     if qwen2_data[i][0] is not None:
#         beta = calculate_adaptive_beta(qwen2_data[i][0])
#         qwen2_betas.append(beta)
#         qwen2_values.append(calculate_hl(qwen2_data[i][0], qwen2_data[i][1], beta))
#     else:
#         qwen2_betas.append(np.nan)
#         qwen2_values.append(np.nan)
        
#     if video_llama_data[i][0] is not None:
#         beta = calculate_adaptive_beta(video_llama_data[i][0])
#         video_llama_betas.append(beta)
#         video_llama_values.append(calculate_hl(video_llama_data[i][0], video_llama_data[i][1], beta))
#     else:
#         video_llama_betas.append(np.nan)
#         video_llama_values.append(np.nan)

# # Convert to numpy arrays for easier handling
# qwen25_values = np.array(qwen25_values)
# qwen2_values = np.array(qwen2_values)
# video_llama_values = np.array(video_llama_values)

# # Display values (with NaN handling)
# qwen25_display = [f'{v:.2f}' if not np.isnan(v) else '-' for v in qwen25_values]
# qwen2_display = [f'{v:.2f}' if not np.isnan(v) else '-' for v in qwen2_values]
# video_llama_display = [f'{v:.2f}' if not np.isnan(v) else '-' for v in video_llama_values]

# qwen25_beta_display = [f'β={v:.2f}' if not np.isnan(v) else '-' for v in qwen25_betas]
# qwen2_beta_display = [f'β={v:.2f}' if not np.isnan(v) else '-' for v in qwen2_betas]
# video_llama_beta_display = [f'β={v:.2f}' if not np.isnan(v) else '-' for v in video_llama_betas]

# # Plot setup
# bar_width = 0.25
# x = np.arange(len(categories))
# fig, ax = plt.subplots(figsize=(12, 7))

# # Create bars
# bar1 = ax.bar(x - bar_width, qwen25_values, bar_width, label='Qwen2.5-VL', color='blue', alpha=0.3)
# bar2 = ax.bar(x, qwen2_values, bar_width, label='Qwen2-VL', color='red', alpha=0.3)
# bar3 = ax.bar(x + bar_width, video_llama_values, bar_width, label='VideoLLaMA-3', color='green', alpha=0.3)

# # Add value labels (保持原大小)
# for i, (rect, val) in enumerate(zip(bar1, qwen25_display)):
#     height = rect.get_height() if not np.isnan(rect.get_height()) else 0
#     ax.text(rect.get_x() + rect.get_width()/2., height, f'{val}',
#             ha='center', va='bottom', fontsize=22, fontfamily='Times New Roman', fontweight='bold')

# for i, (rect, val) in enumerate(zip(bar2, qwen2_display)):
#     height = rect.get_height() if not np.isnan(rect.get_height()) else 0
#     ax.text(rect.get_x() + rect.get_width()/2., height, f'{val}',
#             ha='center', va='bottom', fontsize=22, fontfamily='Times New Roman', fontweight='bold')

# for i, (rect, val) in enumerate(zip(bar3, video_llama_display)):
#     if val != '-':  # Skip NaN value
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width()/2., height, f'{val}',
#                 ha='center', va='bottom', fontsize=22, fontfamily='Times New Roman', fontweight='bold')

# # Customize plot with larger fonts, bold weight and rotated x-axis labels
# # 其他字体再放大两号（从原来的基础上再放大两号）
# ax.set_xticks(x)
# ax.set_xticklabels(categories, fontsize=22, rotation=45, ha='right', fontfamily='Times New Roman', fontweight='bold')
# ax.set_ylim(0, 1.25)
# ax.tick_params(axis='y', labelsize=22, direction='in', width=2, length=6)
# ax.set_xlabel('Fine-Tuning Strategy', fontsize=24, fontfamily='Times New Roman', fontweight='bold')
# ax.set_ylabel('Logic Hallucination Level (HL)', fontsize=24, fontfamily='Times New Roman', fontweight='bold')
# ax.set_title('Adaptive Logic Hallucination Measurement (Adaptive β)', fontsize=26, fontfamily='Times New Roman', fontweight='bold')

# # 将图例放在图表上方，并横向排列
# legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03), 
#                   ncol=3, prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 22})

# ax.grid(axis='y', alpha=0.5)

# # Adjust layout to accommodate rotated labels and legend
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为图例预留空间

# # Save or show plot
# plt.savefig('logic_hallucination_adaptive_formula.png', dpi=300, bbox_inches='tight')
# plt.show()

# # 打印详细信息
# print("Detailed Results with Adaptive Beta:")
# print("=" * 50)
# print("Qwen2.5-VL:")
# for i, cat in enumerate(categories):
#     if not np.isnan(qwen25_values[i]):
#         print(f"  {cat}: HL={qwen25_display[i]}, β={qwen25_beta_display[i]}")

# print("\nQwen2-VL:")
# for i, cat in enumerate(categories):
#     if not np.isnan(qwen2_values[i]):
#         print(f"  {cat}: HL={qwen2_display[i]}, β={qwen2_beta_display[i]}")

# print("\nVideo-LLaMA3:")
# for i, cat in enumerate(categories):
#     if not np.isnan(video_llama_values[i]):
#         print(f"  {cat}: HL={video_llama_display[i]}, β={video_llama_beta_display[i]}")
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体为Times New Roman并加粗
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'

# Data setup
categories = ['No Fine-Tuning', 'Supervised', 'QLoRA 4-bit', 'QLoRA 8-bit', 'After VDAT']

# Original data (ori_correct, sub_correct)
qwen25_data = [(0.6226, 0.5155), (0.6407, 0.6119), (0.7028, 0.6223), (0.7028, 0.6234), (0.5729, 0.5476), (0.6667, 0.6705), (0.7667, 0.6667)]
qwen2_data = [(0.5990, 0.6797), (0.6594, 0.7627), (0.6475, 0.7095), (0.6418, 0.7141), (0.6633, 0.7451), (0.7691, 0.7417), (0.8116, 0.8023)]
video_llama_data = [(0.7752, 0.7400), (0.7349, 0.7812), (None, None), (None, None), (0.7810, 0.7521), (0.8043, 0.7801), (None, None)]

# 动态计算beta值的函数
def calculate_adaptive_beta(ori_correct, base_beta=0.5, threshold=0.7):
    """
    根据主问题正确率动态计算beta值
    - 当ori_correct > threshold时，beta偏向0（更关注相对性能）
    - 当ori_correct < threshold时，beta偏向1（更关注绝对性能）
    """
    if ori_correct is None:
        return base_beta
    
    # 使用sigmoid函数来平滑过渡
    # 当ori_correct增加时，beta减小，更关注相对性能
    adaptive_factor = 1 / (1 + np.exp(5 * (ori_correct - threshold)))
    beta = base_beta + 0.3 * (0.5 - adaptive_factor)
    
    # 限制beta在合理范围内
    return np.clip(beta, 0.2, 0.8)

# Calculate HL values using adaptive beta
def calculate_hl(ori_correct, sub_correct, beta=None):
    if ori_correct is None or ori_correct == 0:
        return np.nan
    
    # 如果没有提供beta，则使用自适应计算
    if beta is None:
        beta = calculate_adaptive_beta(ori_correct)
    
    # HL = S_i * A_i^(beta-1)
    return sub_correct * (ori_correct ** (beta - 1))

# Calculate values for each model with adaptive beta
qwen25_values = []
qwen2_values = []
video_llama_values = []

qwen25_betas = []
qwen2_betas = []
video_llama_betas = []

# Map to display names: No Fine-Tuning, Supervised, QLoRA 4-bit Full, QLoRA 8-bit Full, After VDAT
# We only display 5 categories in the chart, so we select appropriate data points
display_indices = [0, 1, 4, 5, 6]

for i in display_indices:
    if qwen25_data[i][0] is not None:
        beta = calculate_adaptive_beta(qwen25_data[i][0])
        qwen25_betas.append(beta)
        qwen25_values.append(calculate_hl(qwen25_data[i][0], qwen25_data[i][1], beta))
    else:
        qwen25_betas.append(np.nan)
        qwen25_values.append(np.nan)
        
    if qwen2_data[i][0] is not None:
        beta = calculate_adaptive_beta(qwen2_data[i][0])
        qwen2_betas.append(beta)
        qwen2_values.append(calculate_hl(qwen2_data[i][0], qwen2_data[i][1], beta))
    else:
        qwen2_betas.append(np.nan)
        qwen2_values.append(np.nan)
        
    if video_llama_data[i][0] is not None:
        beta = calculate_adaptive_beta(video_llama_data[i][0])
        video_llama_betas.append(beta)
        video_llama_values.append(calculate_hl(video_llama_data[i][0], video_llama_data[i][1], beta))
    else:
        video_llama_betas.append(np.nan)
        video_llama_values.append(np.nan)

# Convert to numpy arrays for easier handling
qwen25_values = np.array(qwen25_values)
qwen2_values = np.array(qwen2_values)
video_llama_values = np.array(video_llama_values)

# Display values (with NaN handling)
qwen25_display = [f'{v:.2f}' if not np.isnan(v) else '-' for v in qwen25_values]
qwen2_display = [f'{v:.2f}' if not np.isnan(v) else '-' for v in qwen2_values]
video_llama_display = [f'{v:.2f}' if not np.isnan(v) else '-' for v in video_llama_values]

qwen25_beta_display = [f'β={v:.2f}' if not np.isnan(v) else '-' for v in qwen25_betas]
qwen2_beta_display = [f'β={v:.2f}' if not np.isnan(v) else '-' for v in qwen2_betas]
video_llama_beta_display = [f'β={v:.2f}' if not np.isnan(v) else '-' for v in video_llama_betas]

# Plot setup
bar_width = 0.25
x = np.arange(len(categories))
fig, ax = plt.subplots(figsize=(12, 7))

# Create bars with different grayscale patterns
bar1 = ax.bar(x - bar_width, qwen25_values, bar_width, label='Qwen2.5-VL', 
              color='black', alpha=0.3, hatch='///')  # 使用斜线阴影
bar2 = ax.bar(x, qwen2_values, bar_width, label='Qwen2-VL', 
              color='black', alpha=0.6)  # 使用点状阴影
bar3 = ax.bar(x + bar_width, video_llama_values, bar_width, label='VideoLLaMA-3', 
              color='black', alpha=0.9)  # 使用垂直线阴影

# Add value labels (保持原大小)
for i, (rect, val) in enumerate(zip(bar1, qwen25_display)):
    height = rect.get_height() if not np.isnan(rect.get_height()) else 0
    ax.text(rect.get_x() + rect.get_width()/2., height, f'{val}',
            ha='center', va='bottom', fontsize=22, fontfamily='Times New Roman', fontweight='bold')

for i, (rect, val) in enumerate(zip(bar2, qwen2_display)):
    height = rect.get_height() if not np.isnan(rect.get_height()) else 0
    ax.text(rect.get_x() + rect.get_width()/2., height, f'{val}',
            ha='center', va='bottom', fontsize=22, fontfamily='Times New Roman', fontweight='bold')

for i, (rect, val) in enumerate(zip(bar3, video_llama_display)):
    if val != '-':  # Skip NaN value
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height, f'{val}',
                ha='center', va='bottom', fontsize=22, fontfamily='Times New Roman', fontweight='bold')

# Customize plot with larger fonts, bold weight and rotated x-axis labels
# 其他字体再放大两号（从原来的基础上再放大两号）
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=22, rotation=45, ha='right', fontfamily='Times New Roman', fontweight='bold')
ax.set_ylim(0, 1.25)
ax.tick_params(axis='y', labelsize=22, direction='in', width=2, length=6)
ax.set_xlabel('Fine-Tuning Strategy', fontsize=24, fontfamily='Times New Roman', fontweight='bold')
ax.set_ylabel('Logic Hallucination Level (HL)', fontsize=24, fontfamily='Times New Roman', fontweight='bold')
ax.set_title('Adaptive Logic Hallucination Measurement (Adaptive β)', fontsize=26, fontfamily='Times New Roman', fontweight='bold')

# 将图例放在图表上方，并横向排列
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03), 
                  ncol=3, prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 22})

ax.grid(axis='y', alpha=0.5)

# Adjust layout to accommodate rotated labels and legend
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为图例预留空间

# Save or show plot
plt.savefig('logic_hallucination_adaptive_formula.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印详细信息
print("Detailed Results with Adaptive Beta:")
print("=" * 50)
print("Qwen2.5-VL:")
for i, cat in enumerate(categories):
    if not np.isnan(qwen25_values[i]):
        print(f"  {cat}: HL={qwen25_display[i]}, β={qwen25_beta_display[i]}")

print("\nQwen2-VL:")
for i, cat in enumerate(categories):
    if not np.isnan(qwen2_values[i]):
        print(f"  {cat}: HL={qwen2_display[i]}, β={qwen2_beta_display[i]}")

print("\nVideo-LLaMA3:")
for i, cat in enumerate(categories):
    if not np.isnan(video_llama_values[i]):
        print(f"  {cat}: HL={video_llama_display[i]}, β={video_llama_beta_display[i]}")