import json
import re
import argparse

def extract_answer(pred_response):
    match = re.search(r"<answer>(.*?)</answer>", pred_response)
    if match:
        return match.group(1).strip()
    return pred_response

def process_file(input_path, output_path):
    # 初始化统计变量
    stats = {
        "0": {"total": 0, "correct": 0},
        "1": {"total": 0, "correct": 0}
    }

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line.strip())

            # 只处理 type == 0 或 type == 1 的条目
            item_type = data.get("type")
            if item_type not in ["0", "1"]:
                pred_response = data.get("pred_response", "")
                pred_ans = extract_answer(pred_response)
                data["pred_response"] = pred_ans
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            pred_response = data.get("pred_response", "")
            pred_ans = extract_answer(pred_response)
            ans = data.get("ans", None)

            # 更新 pred_ans 和 is_correct
            if pred_ans is not None:
                data["pred_ans"] = pred_ans
                data["is_correct"] = (pred_ans == ans)
            else:
                data["pred_ans"] = "Unknown"
                data["is_correct"] = False

            # 统计
            stats[item_type]["total"] += 1
            if data["is_correct"]:
                stats[item_type]["correct"] += 1

            # 写入修改后的行
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

    # 计算并输出正确率
    for t in ["0", "1"]:
        total = stats[t]["total"]
        correct = stats[t]["correct"]
        accuracy = correct / total if total > 0 else 0
        print(f"Type {t} 正确率: {correct}/{total} = {accuracy:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="/root/autodl-tmp/Video-R1/results/Video-R1/prediction_long_result.jsonl")
    parser.add_argument("--output", type=str,
                        default="/root/autodl-tmp/Video-R1/results/video-r1-r/prediction_long_result.jsonl")
    args = parser.parse_args()

    process_file(args.input, args.output)
    print(f"处理完成，结果已保存至: {args.output}")