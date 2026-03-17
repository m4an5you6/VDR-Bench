import numpy as np
import sys

# Ensure proper encoding for output
sys.stdout.reconfigure(encoding='utf-8')

# Original data from the table
# Format: (ori_correct, sub_correct)
data = {
    'Qwen2.5-VL_No_FT': (0.6226, 0.5155),
    'Qwen2.5-VL_Supervised': (0.6407, 0.6119),
    'Qwen2.5-VL_QLoRA_4bit_Full': (0.5729, 0.5476),
    'Qwen2.5-VL_QLoRA_8bit_Full': (0.6667, 0.6705),
    'Qwen2.5-VL_After_VDAT': (0.7667, 0.6667),
    
    'Qwen2-VL_No_FT': (0.5990, 0.6797),
    'Qwen2-VL_Supervised': (0.6594, 0.7627),
    'Qwen2-VL_QLoRA_4bit_Full': (0.6633, 0.7451),
    'Qwen2-VL_QLoRA_8bit_Full': (0.7691, 0.7417),
    'Qwen2-VL_After_VDAT': (0.8116, 0.8023),
    
    'Video-LLaMA3_No_FT': (0.7752, 0.7400),
    'Video-LLaMA3_Supervised': (0.7349, 0.7812),
    'Video-LLaMA3_QLoRA_4bit_Full': (0.7810, 0.7521),
    'Video-LLaMA3_QLoRA_8bit_Full': (0.8043, 0.7801),
}

# Formula 1: HL = (S_i/A_i) / (1 + e^(-k(A_i - A_threshold)))
def hl_formula1(ori_correct, sub_correct, k=10, a_threshold=0.7):
    if ori_correct == 0:
        return 0
    ratio = sub_correct / ori_correct
    denominator = 1 + np.exp(-k * (ori_correct - a_threshold))
    return ratio / denominator

# Formula 2: HL = (S_i/A_i) * (1 - alpha + alpha * A_i)
def hl_formula2(ori_correct, sub_correct, alpha=0.5):
    if ori_correct == 0:
        return 0
    ratio = sub_correct / ori_correct
    return ratio * (1 - alpha + alpha * ori_correct)

# Formula 3: HL = (S_i/A_i) * A_i^beta = S_i * A_i^(beta-1)
def hl_formula3(ori_correct, sub_correct, beta=0.5):
    if ori_correct == 0:
        return 0
    return sub_correct * (ori_correct ** (beta - 1))

# Calculate values for each formula
print("Comparison of HL Formulas")
print("=" * 80)
print(f"{'Model':<25} {'ori/sub':<8} {'HL_Formula1':<12} {'HL_Formula2':<12} {'HL_Formula3':<12}")
print("-" * 80)

results = {}
for model, (ori_correct, sub_correct) in data.items():
    ori_sub_ratio = sub_correct / ori_correct
    hl1 = hl_formula1(ori_correct, sub_correct)
    hl2 = hl_formula2(ori_correct, sub_correct)
    hl3 = hl_formula3(ori_correct, sub_correct)
    
    results[model] = {
        'ori_sub': ori_sub_ratio,
        'hl1': hl1,
        'hl2': hl2,
        'hl3': hl3
    }
    
    print(f"{model:<25} {ori_sub_ratio:<8.4f} {hl1:<12.4f} {hl2:<12.4f} {hl3:<12.4f}")

# Calculate improvement for fine-tuned models compared to no fine-tuning
print("\n" + "=" * 80)
print("Improvement Analysis (Fine-tuned vs No Fine-tuning)")
print("=" * 80)

models = ['Qwen2.5-VL', 'Qwen2-VL', 'Video-LLaMA3']
formulas = [
    ("HL Formula 1", lambda ori, sub: hl_formula1(ori, sub)),
    ("HL Formula 2", lambda ori, sub: hl_formula2(ori, sub)),
    ("HL Formula 3", lambda ori, sub: hl_formula3(ori, sub))
]

for model in models:
    no_ft_key = f'{model}_No_FT'
    supervised_key = f'{model}_Supervised'
    
    no_ft_ori, no_ft_sub = data[no_ft_key]
    supervised_ori, supervised_sub = data[supervised_key]
    
    print(f"\n{model}:")
    print(f"  No Fine-tuning: ori={no_ft_ori:.4f}, sub={no_ft_sub:.4f}, ori/sub={no_ft_sub/no_ft_ori:.4f}")
    print(f"  Supervised FT:  ori={supervised_ori:.4f}, sub={supervised_sub:.4f}, ori/sub={supervised_sub/supervised_ori:.4f}")
    
    for formula_name, formula_func in formulas:
        no_ft_hl = formula_func(no_ft_ori, no_ft_sub)
        supervised_hl = formula_func(supervised_ori, supervised_sub)
        improvement = ((supervised_hl - no_ft_hl) / no_ft_hl) * 100 if no_ft_hl != 0 else float('inf')
        print(f"    {formula_name}: {no_ft_hl:.4f} -> {supervised_hl:.4f} ({'+' if improvement >= 0 else ''}{improvement:.2f}%)")

# Save results to file for further analysis
with open('hl_formula_results.txt', 'w', encoding='utf-8') as f:
    f.write("Comparison of HL Formulas\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'Model':<25} {'ori/sub':<8} {'HL_Formula1':<12} {'HL_Formula2':<12} {'HL_Formula3':<12}\n")
    f.write("-" * 80 + "\n")
    
    for model, (ori_correct, sub_correct) in data.items():
        ori_sub_ratio = sub_correct / ori_correct
        hl1 = hl_formula1(ori_correct, sub_correct)
        hl2 = hl_formula2(ori_correct, sub_correct)
        hl3 = hl_formula3(ori_correct, sub_correct)
        f.write(f"{model:<25} {ori_sub_ratio:<8.4f} {hl1:<12.4f} {hl2:<12.4f} {hl3:<12.4f}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("Improvement Analysis (Fine-tuned vs No Fine-tuning)\n")
    f.write("=" * 80 + "\n")
    
    for model in models:
        no_ft_key = f'{model}_No_FT'
        supervised_key = f'{model}_Supervised'
        
        no_ft_ori, no_ft_sub = data[no_ft_key]
        supervised_ori, supervised_sub = data[supervised_key]
        
        f.write(f"\n{model}:\n")
        f.write(f"  No Fine-tuning: ori={no_ft_ori:.4f}, sub={no_ft_sub:.4f}, ori/sub={no_ft_sub/no_ft_ori:.4f}\n")
        f.write(f"  Supervised FT:  ori={supervised_ori:.4f}, sub={supervised_sub:.4f}, ori/sub={supervised_sub/supervised_ori:.4f}\n")
        
        for formula_name, formula_func in formulas:
            no_ft_hl = formula_func(no_ft_ori, no_ft_sub)
            supervised_hl = formula_func(supervised_ori, supervised_sub)
            improvement = ((supervised_hl - no_ft_hl) / no_ft_hl) * 100 if no_ft_hl != 0 else float('inf')
            f.write(f"    {formula_name}: {no_ft_hl:.4f} -> {supervised_hl:.4f} ({'+' if improvement >= 0 else ''}{improvement:.2f}%)\n")

print(f"\nDetailed results saved to 'hl_formula_results.txt'")