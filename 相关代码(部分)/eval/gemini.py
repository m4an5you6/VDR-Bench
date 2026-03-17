import argparse
import base64
import json
import os
from tqdm import tqdm
from zhipuai import ZhipuAI
import random
import string

# 用于存储已生成的 id
generated_ids = set()

# 生成唯一的 8 位随机字符串
def generate_unique_id():
    while True:
        new_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        if new_id not in generated_ids:  # 检查是否已存在
            generated_ids.add(new_id)  # 添加到集合中
            return new_id

# 读取视频文件并转换为 base64 编码
def load_video_as_base64(video_path):
    with open(video_path, 'rb') as video_file:
        video_base = base64.b64encode(video_file.read()).decode('utf-8')
    return video_base

# 调用模型生成回答
def generate_response(video_base, prompt):
    client = ZhipuAI(api_key="12aa9ae12694fee22796db693d6065bd.gB289tk8e9DBzAzc")  # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4v-plus",  # 填写需要调用的模型名称
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": video_base
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content

# 主函数
def eval_model(args):
    # 读取 JSONL 文件
    questions = []
    with open(os.path.expanduser(args.question_file), "r") as f:
        for eachline in f:
            questions.append(json.loads(eachline))

    # 创建输出文件
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # 用于保存 caption
    caption_file = open("/root/autodl-tmp/caption_eval/bert_score-master/referrence.txt", "w")
    caption_answer_file = open("/root/autodl-tmp/caption_eval/bert_score-master/model_predict.txt", "w")

    # 用于统计准确率
    total_ori_qa = 0
    correct_ori_qa = 0
    total_sub_qa = 0
    correct_sub_qa = 0

    # 遍历每个问题
    for line in tqdm(questions, desc="Processing"):
        # 生成唯一的 id
        unique_id = generate_unique_id()
        src_dataset = line["src_dataset"]
        video_file = line["video_name"]
        video_path = os.path.join(args.video_folder, src_dataset, video_file)

        # 读取视频并转换为 base64
        video_base = load_video_as_base64(video_path)

        # 第一部分：处理 original_qa
        original_qa = line["original_qa"]
        qs = original_qa["qs"]
        choices_str = ', '.join([f"{key}: {value}" for key, value in original_qa["choice"].items()])
        qs = f"Question: {qs}, Choices: {choices_str}. Answer me using only the given options (A, B, C...)"

        # 生成回答
        response = generate_response(video_base, qs)

        # 统计 ori_qa 准确率
        total_ori_qa += 1
        if response.strip().upper() == original_qa["ans"].strip().upper():
            correct_ori_qa += 1

        # 保存 original_qa 结果
        result_data = {
            "id": unique_id,
            "type": "0",
            "src_dataset": src_dataset,
            "video_name": video_file,
            "prompt": qs,
            "pred_response": response,
            "ans": original_qa["ans"],
            "aspect": line["aspect"],
        }
        ans_file.write(json.dumps(result_data) + "\n")
        ans_file.flush()

        # 第二部分：处理 sub_qa
        sub_qas = line["sub_qas"]
        for sub_qa_key, sub_qa in sub_qas.items():
            qs = sub_qa["qs"]
            choices_str = ', '.join([f"{key}: {value}" for key, value in sub_qa["choice"].items()])
            qs = f"Question: {qs}, Choices: {choices_str}. Answer me using only the given options (A, B, C...)"

            # 生成回答
            response = generate_response(video_base, qs)

            # 统计 sub_qa 准确率
            total_sub_qa += 1
            if response.strip().upper() == sub_qa["ans"].strip().upper():
                correct_sub_qa += 1

            # 保存 sub_qa 结果
            result_data = {
                "id": unique_id,
                "type": "1",
                "src_dataset": src_dataset,
                "video_name": video_file,
                "prompt": qs,
                "pred_response": response,
                "ans": sub_qa["ans"],
                "aspect": line["aspect"],
            }
            ans_file.write(json.dumps(result_data) + "\n")
            ans_file.flush()

        # 第三部分：处理 caption
        caption_prompt = "Generate a comprehensive and concise description of the entire video, summarizing its key activities and events without focusing on individual frames."

        # 生成回答
        response = generate_response(video_base, caption_prompt)

        # 保存 caption 结果
        result_data = {
            "id": unique_id,
            "type": "2",
            "src_dataset": src_dataset,
            "video_name": video_file,
            "prompt": caption_prompt,
            "pred_response": response,
            "ans": line["caption"],
            "aspect": line["aspect"],
        }
        ans_file.write(json.dumps(result_data) + "\n")
        ans_file.flush()

        # 去除换行符并保存 caption
        cleaned_caption = response.replace("\n", " ").strip()
        caption_file.write(cleaned_caption + "\n")
        caption_answer_file.write(line["caption"] + "\n")

    # 关闭文件
    ans_file.close()
    caption_file.close()
    caption_answer_file.close()

    # 计算并输出准确率
    ori_qa_accuracy = (correct_ori_qa / total_ori_qa) * 100 if total_ori_qa > 0 else 0
    sub_qa_accuracy = (correct_sub_qa / total_sub_qa) * 100 if total_sub_qa > 0 else 0
    print(f"Original QA Accuracy: {ori_qa_accuracy:.2f}%")
    print(f"Sub QA Accuracy: {sub_qa_accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="/root/autodl-tmp/pace.jsonl")
    parser.add_argument("--answers-file", type=str, default="/root/autodl-tmp/results/gemini/answer_pace.jsonl")
    parser.add_argument("--video-folder", type=str, default="/root/autodl-tmp/data")
    args = parser.parse_args()

    eval_model(args)