import argparse
import os
import cv2
import json
from tqdm import tqdm
import random
import string
import torch
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import random
import string
import glob
# 用于存储已生成的 id
generated_ids = set()
def find_video_file(folder_path, target_filename):
    """
    遍历文件夹及其子文件夹，查找名为 target_filename 的视频文件。
    :param folder_path: 文件夹路径
    :param target_filename: 要查找的目标文件名，默认为 4.mp4
    :return: 找到的文件路径，若未找到则返回空列表
    """
    
    # 遍历目录及子目录
    for root, dirs, files in os.walk(folder_path):
        # 检查是否有目标文件
        if target_filename in files:
            # 将文件的完整路径加入到 found_files 列表
            return os.path.join(root, target_filename)
# 生成唯一的 8 位随机字符串
def generate_unique_id():
    while True:
        new_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        if new_id not in generated_ids:  # 检查是否已存在
            generated_ids.add(new_id)  # 添加到集合中
            return new_id
def get_video_duration(video_path, default_fps=16):
    cap = cv2.VideoCapture(video_path)
    print(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 如果 fps 为零，使用默认帧率
    if fps == 0:
        fps = default_fps
        print(f"Warning: FPS is zero, using default FPS: {default_fps}")
    
    duration = frame_count / fps
    cap.release()
    return duration, fps

# 主函数
def eval_model(args):
    # 加载模型
    #device = torch.device("cuda:0")  # 使用第一张 GPU
    #args.model_path,
    model_path1 = "/root/autodl-tmp/qwen_finetune/lmms-finetune-main/checkpoints/qwen2-vl-7b-instruct_lora-True_qlora-False"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    #lora_path = "/root/autodl-tmp/qwen_finetune/lmms-finetune-main/checkpoints/qwen2-vl-7b-instruct_lora-True_qlora-False/checkpoint-8"  # 替换为你的 LoRA 路径
    lora_path = "/root/autodl-tmp/qwen_finetune/lmms-finetune-main/checkpoints/qwen2-vl-2b-instruct_lora-True_qlora-False"  # 替换为你的 LoRA 路径
    model = PeftModel.from_pretrained(model, lora_path)
    #processor = Qwen2VLProcessor.from_pretrained(args.model_path)
    processor = Qwen2VLProcessor.from_pretrained(args.model_path)
    for name, param in model.named_parameters():
        if "lora" in name:
            print(name, param)
    model.eval()
    #model.to(device)  # 将模型移动到指定设备
    # 创建输出文件
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # 读取 JSONL 文件
    questions = []
    with open(os.path.expanduser(args.question_file), "r") as f:
        for eachline in f:
            questions.append(json.loads(eachline))

    # 用于统计准确率
    total_ori_qa = 0
    correct_ori_qa = 0
    total_sub_qa = 0
    correct_sub_qa = 0

    # 用于保存 caption
    caption_file = open("/root/autodl-tmp/caption_eval/bert_score-master/referrence.txt", "w")
    caption_answer_file = open("/root/autodl-tmp/caption_eval/bert_score-master/model_predict.txt", "w")

    # 遍历每个问题
    for line in tqdm(questions, desc="Qwen eval"):
        unique_id = generate_unique_id()
        src_dataset = line["src_dataset"]
        video_file = line["video_name"]
        video_path=find_video_file(args.video_folder, video_file)
        #print(video_path)
        #video_path = os.path.join(args.video_folder, video_file)
        #print(video_path)

        # 获取视频时长
        duration, fps = get_video_duration(video_path)

        # 第一部分：处理 original_qa
        original_qa = line["original_qa"]
        qs = original_qa["qs"]
        choices_str = ', '.join([f"{key}: {value}" for key, value in original_qa["choice"].items()])
        qs = f"Question: {qs}, Choices: {choices_str}. Answer me using only the given options (A, B, C...)"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": args.for_get_frames_num / duration,
                    },
                    {"type": "text", "text": qs},
                ],
            }
        ]

        with torch.inference_mode():
            # 准备推理
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # 推理
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        # 统计 ori_qa 准确率
        total_ori_qa += 1
        if output_text[0].strip().upper() == original_qa["ans"].strip().upper():
            correct_ori_qa += 1

        # 保存 original_qa 结果
        result_data = {
            "video_id": unique_id,
            "type": "0",
            "src_dataset": src_dataset,
            "video_name": video_file,
            "prompt": qs,
            "pred_response": output_text[0],
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

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": 360 * 420,
                            "fps": args.for_get_frames_num / duration,
                        },
                        {"type": "text", "text": qs},
                    ],
                }
            ]

            with torch.inference_mode():
                # 准备推理
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")

                # 推理
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

            # 统计 sub_qa 准确率
            total_sub_qa += 1
            if output_text[0].strip().upper() == sub_qa["ans"].strip().upper():
                correct_sub_qa += 1

            # 保存 sub_qa 结果
            result_data = {
                "video_id": unique_id,
                "type": "1",
                "src_dataset": src_dataset,
                "video_name": video_file,
                "prompt": qs,
                "pred_response": output_text[0],
                "ans": sub_qa["ans"],
                "aspect": line["aspect"],
            }
            ans_file.write(json.dumps(result_data) + "\n")
            ans_file.flush()

        # 第三部分：处理 caption
        caption_prompt = "Generate a comprehensive and concise description of the entire video, summarizing its key activities and events without focusing on individual frames. Avoid repeating the same sentence or phrase."

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": args.for_get_frames_num / duration,
                    },
                    {"type": "text", "text": caption_prompt},
                ],
            }
        ]

        with torch.inference_mode():
            # 准备推理
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # 推理
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        # 保存 caption 结果
        result_data = {
            "video_id": unique_id,
            "type": "2",
            "src_dataset": src_dataset,
            "video_name": video_file,
            "prompt": caption_prompt,
            "pred_response": output_text[0],
            "ans": line["caption"],
            "aspect": line["aspect"],
        }
        ans_file.write(json.dumps(result_data) + "\n")
        ans_file.flush()

        # 去除换行符并保存 caption
        cleaned_caption = output_text[0].replace("\n", " ").strip()
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
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--model-path_finetune", type=str, default="/root/autodl-tmp/qwen_finetune/lmms-finetune-main/checkpoints/qwen2-vl-7b-instruct_lora-True_qlora-False/")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="/root/autodl-tmp/validation_set.jsonl")
    parser.add_argument("--answers-file", type=str, default="/root/autodl-tmp/results/Qwen2-VL-2B-Instruct/origin.jsonl")
    #parser.add_argument("--video-folder", type=str, default="/root/autodl-tmp/data/reason/short/root/autodl-tmp/reason/short")
    parser.add_argument("--video-folder", type=str, default="/root/autodl-tmp/data")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--for_get_frames_num", type=int, default=16)
    args = parser.parse_args()

    eval_model(args)