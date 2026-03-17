import json

# 从文件中读取 JSON 数据
input_file = '/root/autodl-tmp/results/VILA1.5-7b/sequence_short.jsonl'  # 输入文件名
pred_output_file = 'model_predict.txt'  # pred_response 输出文件名
ans_output_file = 'referrence.txt'  # ans 输出文件名

# 打开输出文件
with open(pred_output_file, 'w', encoding='utf-8') as pred_f, \
     open(ans_output_file, 'w', encoding='utf-8') as ans_f:
    
    # 逐行读取 JSONL 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析每一行的 JSON 数据
            try:
                data = json.loads(line.strip())
                
                # 检查 type 是否为 2
                if data.get('type') == "2":
                    # 提取 pred_response 和 ans，并去除内部换行符
                    pred_response = data.get('pred_response', '').replace("\n", " ").strip()
                    ans = data.get('ans', '').replace("\n", " ").strip()
                    
                    # 写入 pred_response 到 pred_response.txt（非空内容才写入）
                    if pred_response:
                        pred_f.write(pred_response + '\n')
                    
                    # 写入 ans 到 ans.txt（非空内容才写入）
                    if ans:
                        ans_f.write(ans + '\n')
            except json.JSONDecodeError:
                print(f"JSON 解析错误: {line.strip()}")

print(f"pred_response 已写入 {pred_output_file}")
print(f"ans 已写入 {ans_output_file}")