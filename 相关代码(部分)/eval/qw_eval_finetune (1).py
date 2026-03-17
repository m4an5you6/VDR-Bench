import argparse
import os
import cv2
import json
from tqdm import tqdm
import random
import string
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import random
import string
import subprocess
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 用于存储已生成的 id
generated_ids = set()
def process_video_path(video_path: str, output_format: str = "mp4") -> str:
    """
    如果视频格式不是 mp4，则将其转换为 mp4 并保存在相同路径下。
    
    Args:
        video_path (str): 原始视频路径。
        output_format (str): 目标格式，默认为 'mp4'。
    
    Returns:
        str: 转换后的视频路径（带 .mp4 后缀）。
    """
    base, ext = os.path.splitext(video_path)
    
    # 如果已经是目标格式，直接返回原路径
    if ext.lower() == f".{output_format}":
        logger.info(f"Video is already in .{output_format} format: {video_path}")
        return video_path
    
    # 构建输出路径
    output_video_path = f"{base}.{output_format}"
    
    # 如果已存在转换后的文件，跳过转换
    if os.path.exists(output_video_path):
        logger.info(f"Converted video already exists: {output_video_path}")
        return output_video_path

    return output_video_path
# 生成唯一的 8 位随机字符串
def generate_unique_id():
    while True:
        new_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        if new_id not in generated_ids:  # 检查是否已存在
            generated_ids.add(new_id)  # 添加到集合中
            return new_id
def get_video_duration(video_path, default_fps=16):
    cap = cv2.VideoCapture(video_path)
    #print(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 如果 fps 为零，使用默认帧率
    if fps == 0:
        fps = default_fps
        print(f"Warning: FPS is zero, using default FPS: {default_fps}")
    
    duration = frame_count / fps
    cap.release()
    return duration, fps

def search_video_by_name(start_folder, target_filename):
    """
    在指定文件夹及子文件夹中查找指定文件名的视频文件
    :param start_folder: 要搜索的起始文件夹
    :param target_filename: 要查找的文件名（含扩展名，如 "video.mp4"）
    :return: 匹配的视频文件路径列表
    """
    VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm'}
    matched_videos = []

    for root, dirs, files in os.walk(start_folder):
        for file in files:
            name, ext = os.path.splitext(file)
            full_name = file  # 带扩展名的文件名
            if ext.lower() in VIDEO_EXTENSIONS and full_name == target_filename:
                matched_videos.append(os.path.join(root, file))

    return matched_videos
# 主函数
def eval_model(args):
    # 加载模型
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     args.model_path,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto"
    # )
    #model = Qwen2_5_VLForConditionalGeneration.from_pretrained("/root/autodl-tmp/huggingface/hub/qwen2.5vl-ft-100")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/root/autodl-tmp/huggingface/hub/qwen2.5vl-ft_14000",
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    local_files_only=True, 
)
        # 检查模型参数是否包含 NaN
    processor = AutoProcessor.from_pretrained(args.model_path)
    from peft import PeftModel, PeftConfig

    # 加载基础模型
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     args.model_path,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    #     attn_implementation="flash_attention_2",
    # )
    
    #加载 LoRA 适配器
    #model = PeftModel.from_pretrained(model, "/root/autodl-tmp/LLaMA-Factory/saves/qwen2_5vl-7b/lora/sft_8000_10000")
    for name, param in model.named_parameters():
        print(param)
        if torch.isnan(param).any():
            print(f"Warning: Parameter '{name}' contains NaN!")
    # 合并模型
    # merged_model = lora_model.merge_and_unload()
    
    # # 检查模型参数是否包含 NaN
    # for name, param in merged_model.named_parameters():
    #     if torch.isnan(param).any():
    #         print(f"Warning: Parameter '{name}' contains NaN!")
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
    # 遍历每个问题
    for line in tqdm(questions, desc="Qwen eval"):
        line = json.loads(line.strip())
        unique_id = generate_unique_id()
        src_dataset = line["src_dataset"]
        video_file = line["video_name"]
        if(video_file=="0PVKV.mp4"):
            continue
        # video_path = os.path.join(args.video_folder, args.model_aspect)
        # video_path = os.path.join(video_path, video_file)
        video_path=search_video_by_name(args.video_folder,video_file)[0]
        #print(video_path,video_file)
        video_path =process_video_path(video_path)
        # 获取视频时长
        #logger.info(f"Processing video: {video_path}")
        duration, fps = get_video_duration(video_path)
        #print(f"Duration: {duration}, FPS: {fps}")
        # 第一部分：处理 original_qa
        if(line["type"]=="0"):
            # original_qa = line["original_qa"]
            # qs = original_qa["qs"]
            # choices_str = ', '.join([f"{key}: {value}" for key, value in original_qa["choice"].items()])
            # qs = f"Question: {qs}, Choices: {choices_str}. Answer me using only the given options (A, B, C...)"
            qs=line["prompt"]
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
                #print("aaaaaaaaaaaaaaaaaaa")
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")
    
                # 推理
                generated_ids = model.generate(**inputs, do_sample=True,  temperature=0.7,top_p=0.9, max_new_tokens=args.max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            #     # 准备推理
            # text = processor.apply_chat_template(
            #         messages, tokenize=False, add_generation_prompt=True
            # )
            # print("aaaaaaaaaaaaaaaaaaa")
            # image_inputs, video_inputs = process_vision_info(messages)
            # print("aaaaaaaaaaaaaaaaaaa")
            # inputs = processor(
            #     text=[text],
            #     images=image_inputs,
            #     videos=video_inputs,
            #     padding=True,
            #     return_tensors="pt",
            # )
            # inputs = inputs.to("cuda")
    
            #     # 推理
            # generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            # generated_ids_trimmed = [
            #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            # ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            # 统计 ori_qa 准确率
            total_ori_qa += 1
            if output_text[0].strip().upper() == line["ans"].upper():
                correct_ori_qa += 1
            #print("aaaaaaaaaaaaaaaaaaa")
            # 保存 original_qa 结果
            result_data = {
                "video_id": unique_id,
                "type": "0",
                "src_dataset": src_dataset,
                "video_name": video_file,
                "prompt": qs,
                "pred_response": output_text[0],
                "ans": line["ans"],
                "aspect": line["aspect"],
            }
            ans_file.write(json.dumps(result_data) + "\n")
            ans_file.flush()

        # 第二部分：处理 sub_qa
        if(line["type"]=="1"):
            # qs = sub_qa["qs"]
            # choices_str = ', '.join([f"{key}: {value}" for key, value in sub_qa["choice"].items()])
            # qs = f"Question: {qs}, Choices: {choices_str}. Answer me using only the given options (A, B, C...)"
            qs=line["prompt"]
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
            #print("aaaaaaaaaaaaaaaaaaa")
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
                generated_ids = model.generate(**inputs, do_sample=True,  temperature=0.7,top_p=0.9, max_new_tokens=args.max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

            # 统计 sub_qa 准确率
            total_sub_qa += 1
            if output_text[0].strip().upper() == line["ans"].upper():
                correct_sub_qa += 1
                #print("rrr")
            # 保存 sub_qa 结果
            result_data = {
                "video_id": unique_id,
                "type": "1",
                "src_dataset": src_dataset,
                "video_name": video_file,
                "prompt": qs,
                "pred_response": output_text[0],
                "ans": line["ans"],
                "aspect": line["aspect"],
            }
            ans_file.write(json.dumps(result_data) + "\n")
            ans_file.flush()

    #     # 第三部分：处理 caption
    #     if(line["type"]=="2"):
    #         caption_prompt = "Generate a comprehensive and concise description of the entire video, summarizing its key activities and events without focusing on individual frames. Avoid repeating the same sentence or phrase."
    #         messages = [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "video",
    #                         "video": video_path,
    #                         "max_pixels": 360 * 420,
    #                         "fps": args.for_get_frames_num / duration,
    #                     },
    #                     {"type": "text", "text": caption_prompt},
    #                 ],
    #             }
    #         ]
    
    #         with torch.inference_mode():
    #             # 准备推理
    #             text = processor.apply_chat_template(
    #                 messages, tokenize=False, add_generation_prompt=True
    #             )
    #             image_inputs, video_inputs = process_vision_info(messages)
    #             inputs = processor(
    #                 text=[text],
    #                 images=image_inputs,
    #                 videos=video_inputs,
    #                 padding=True,
    #                 return_tensors="pt",
    #             )
    #             inputs = inputs.to("cuda")
    
    #             # 推理
    #             generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    #             generated_ids_trimmed = [
    #                 out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    #             ]
    #             output_text = processor.batch_decode(
    #                 generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    #             )
    
    #         # 保存 caption 结果
    #         result_data = {
    #             "video_id": unique_id,
    #             "type": "2",
    #             "src_dataset": src_dataset,
    #             "video_name": video_file,
    #             "prompt": caption_prompt,
    #             "pred_response": output_text[0],
    #             "ans": line["ans"],
    #             "aspect": line["aspect"],
    #         }
    #         ans_file.write(json.dumps(result_data) + "\n")
    #         ans_file.flush()
            
    #         # 去除换行符并保存 caption
    #         cleaned_caption = output_text[0].replace("\n", " ").strip()
        
    # # 关闭文件
    # ans_file.close()

    # 计算并输出准确率
    ori_qa_accuracy = (correct_ori_qa / total_ori_qa) * 100 if total_ori_qa > 0 else 0
    sub_qa_accuracy = (correct_sub_qa / total_sub_qa) * 100 if total_sub_qa > 0 else 0
    print(f"Original QA Accuracy: {ori_qa_accuracy:.2f}%")
    print(f"Sub QA Accuracy: {sub_qa_accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model_aspect", type=str, default="sequence_long")
    parser.add_argument("--question-file", type=str, default="/root/autodl-tmp/results/qwen2.5vl-ps-test.jsonl")
    parser.add_argument("--answers-file", type=str, default="/root/autodl-tmp/results/Qwen2.5-VL-7B-Instruct-f/output_14000.jsonl")
    parser.add_argument("--video-folder", type=str, default="/root/autodl-tmp/data")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--for_get_frames_num", type=int, default=16)
    args = parser.parse_args()

    eval_model(args)