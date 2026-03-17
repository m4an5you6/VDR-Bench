import json

def calculate_accuracy(file_path):
    type_correct = {"0": 0, "1": 0}
    type_total = {"0": 0, "1": 0}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
             try:
                data = json.loads(line.strip())
                q_type = data.get("type")
                if q_type not in ["0", "1"]:
                    continue
                is_correct = data.get("is_correct", False)
    
                # 统计总数和正确数
                type_total[q_type] += 1
                if is_correct:
                    type_correct[q_type] += 1
             except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")

    # 计算正确率
    accuracy = {}
    for t in ["0", "1"]:
        total = type_total[t]
        correct = type_correct[t]
        accuracy[t] = correct / total if total > 0 else 0
        print(f"Type {t} Accuracy: {correct}/{total} = {accuracy[t]:.2%}")

    return accuracy

# 示例调用
calculate_accuracy("/root/autodl-tmp/VideoLLaMA3/results/videollama3/pace_long_result.jsonl")