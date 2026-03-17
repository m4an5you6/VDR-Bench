import json

def convert_jsonl_to_target_format(input_jsonl, output_json):
    # 打开输入的 JSONL 文件和输出的 JSON 文件
    with open(input_jsonl, 'r', encoding='utf-8') as infile, open(output_json, 'w', encoding='utf-8') as outfile:
        data = []
        
        # 遍历 JSONL 文件的每一行
        for line in infile:
            record = json.loads(line)
            
            # 提取视频名称
            video_name = record.get('video_name', '')
            
            # 提取描述信息
            caption = record.get('caption', '')
            
            # 构造每个视频的结构
            video_entry = {
                "video": video_name,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<video>Generate a comprehensive and concise description of the entire video, summarizing its key activities and events without focusing on individual frames. Avoid repeating the same sentence or phrase."
                    },
                    {
                        "from": "gpt",
                        "value": caption
                    }
                ]
            }
            
            # 将生成的视频条目添加到数据列表中
            data.append(video_entry)
        
        # 将转换后的数据写入到输出的 JSON 文件（最外层是列表）
        json.dump(data, outfile, ensure_ascii=False, indent=4)

# 调用函数并传入文件路径
input_jsonl = '/root/autodl-tmp/qwen_finetune/lmms-finetune-main/train_df.jsonl'  # 输入的 JSONL 文件路径
output_json = '/root/autodl-tmp/qwen_finetune/lmms-finetune-main/data_trian_delete.json'  # 输出的 JSON 文件路径

convert_jsonl_to_target_format(input_jsonl, output_json)

print(f"转换完成，输出文件：{output_json}")