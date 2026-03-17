import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import math
from decord import VideoReader, cpu
import cv2
import base64
from PIL import Image
import numpy as np
from typing import Dict, List
import re
import time
import traceback
from transformers import AutoModelForCausalLM, AutoProcessor
# 导入正确的VideoLLaMA2模块
from videollama3 import model_init, mm_infer
from videollama3 import disable_torch_init
import warnings
from evaluation.register import INFERENCES
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
import ffmpeg
def load_video(video_path, args):
    """加载视频并采样帧"""
    #print(f"加载视频: {video_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = round(vr.get_avg_fps())
        
        # 采样策略
        frame_idx = [i for i in range(0, len(vr), fps)]
        
        if len(frame_idx) < args.for_get_frames_num:
            frame_idx = np.linspace(0, total_frame_num - 1, args.for_get_frames_num, dtype=int).tolist()
        elif len(frame_idx) > args.for_get_frames_num:
            frame_idx = frame_idx[:args.for_get_frames_num]
        
        frames = vr.get_batch(frame_idx).asnumpy()  # [T, H, W, 3]
        #print(f"视频采样完成: {video_path}, 总帧数: {total_frame_num}, 采样帧数: {len(frames)}")
        return frames
    except Exception as e:
        print(f"加载视频失败: {video_path}, 错误: {str(e)}")
        raise


def format_question_and_choices(question, choices):
    """格式化问题和选项"""
    choices_str = ", ".join([f"{key}: {value}" for key, value in choices.items()])
    return f"Question: {question}, Choices: {choices_str}. Please select the correct answer."


def remove_duplicates_from_caption(caption):
    # 简单去重处理
    return caption


def eval_model(args):
    """评估模型主函数"""
    #disable_torch_init()
    
    # print(f"开始加载模型: {args.model_path}")
    # try:
    #     # 使用示例中的model_init函数加载模型
    #     model, processor, tokenizer = model_init(
    #         args.model_path, 
    #         load_8bit=args.load_8bit, 
    #         load_4bit=args.load_4bit
    #     )
    # except Exception as e:
    #     print(f"模型加载失败: {str(e)}")
    #     traceback.print_exc()
    #     raise
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_path,
    #     trust_remote_code=True,
    #     device_map={"": device},
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    # )
    # processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    #local_rank = int(os.environ["LOCAL_RANK"])
    #model_init, mm_infer = INFERENCES(args.model_path)
    device = "cuda:0"
    model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # model, processor = model_init(
    #     args.model_path,
    #     args.max_visual_tokens,
    #     #device_map={"": f"cuda:{local_rank}"}
    # )
    model = model.to(device)
    model.eval()
    #print(f"模型加载成功, 设备: {device}")
    
    # 验证输出目录
    answers_file = os.path.expanduser(args.answers_file)
    output_dir = os.path.dirname(answers_file)
    if not os.path.exists(output_dir):
        #print(f"创建输出目录: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"输出文件路径: {answers_file}")
    ans_file = open(answers_file, "w")
    
    # 加载问题数据
    question_file = os.path.expanduser(args.question_file)
    #print(f"加载问题数据: {question_file}")
    questions = []
    with open(question_file, "r") as f:
        for line in f:
            questions.append(json.loads(line))
    
    print(f"共加载 {len(questions)} 个样本")
    
    # 计数器初始化
    ori_correct_count = 0
    ori_total_count = 0
    sub_correct_count = 0
    sub_total_count = 0
    
    for line_idx, line in enumerate(tqdm(questions, desc="处理样本")):
        #print(f"\n处理样本 {line_idx+1}/{len(questions)}: {line.get('video_name', '未知')}")
        video_name = line["video_name"]
        video_path = os.path.join(args.video_folder, args.video_aspect)
        video_path = os.path.join(video_path, video_name)
        # 验证视频路径
        if not os.path.exists(video_path):
            print(f"视频文件不存在: {video_path}，跳过该样本")
            continue
        
        # 加载视频
        #video_frames = load_video(video_path, args)
        
        # # 使用processor预处理视频 - 根据示例代码适配
        # processed_video = processor['video'](video_frames).to(device, dtype=torch.float16)
        # print(f"视频预处理完成, shape: {processed_video.shape}")
        
        # 处理原始问题
        original_qa = line["original_qa"]
        qs = format_question_and_choices(original_qa["qs"], original_qa["choice"])
        #print(f"原始问题: {qs[:100]}...")
        
        # 使用示例中的mm_infer函数进行推理
        message = [{'role': 'user', 'content': qs}]
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 16}},
                    {"type": "text", "text":qs},
                ]
            },
        ]
        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output_ids = model.generate(**inputs, max_new_tokens=1024)
        output = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # output = mm_infer(
        #     processed_video, 
        #     message, 
        #     model, 
        #     tokenizer, 
        #     modal='video',
        #     do_sample=True if args.temperature > 0 else False,
        #     temperature=args.temperature,
        #     top_p=args.top_p,
        #     max_new_tokens=args.max_new_tokens
        # )
        
        #print(f"模型回答: {output}")
        
        # 统计正确率
        ori_total_count += 1
        is_correct = original_qa["ans"].lower() in output.lower()
        #print(f"答案是否正确: {is_correct}, 正确答案: {original_qa['ans']}")
        if is_correct:
            ori_correct_count += 1
        
        # 保存结果
        result = {
            "id": video_name,
            "type": "0",
            "src_dataset": line["src_dataset"],
            "video_name": video_name,
            "prompt": qs,
            "pred_response": output,
            "ans": original_qa["ans"],
            "aspect": line["aspect"],
            "is_correct": is_correct
        }
        
        ans_file.write(json.dumps(result) + "\n")
        ans_file.flush()
        
        # 处理子问题
        if "sub_qas" in line and isinstance(line["sub_qas"], dict):
            sub_qas = line["sub_qas"]
            for sub_qa_key, sub_qa in sub_qas.items():
                qs_sub = format_question_and_choices(sub_qa["qs"], sub_qa["choice"])
                #print(f"子问题: {qs_sub[:100]}...")
                #message = [{'role': 'user', 'content': qs}]
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 16}},
                            {"type": "text", "text":qs_sub},
                        ]
                    },
                ]
                inputs = processor(
                    conversation=conversation,
                    add_system_prompt=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                output_ids = model.generate(**inputs, max_new_tokens=1024)
                output_sub = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                # # 使用mm_infer处理子问题
                # message_sub = [{'role': 'user', 'content': qs_sub}]
                # output_sub = mm_infer(
                #     processed_video, 
                #     message_sub, 
                #     model, 
                #     tokenizer, 
                #     modal='video',
                #     do_sample=True if args.temperature > 0 else False,
                #     temperature=args.temperature,
                #     top_p=args.top_p,
                #     max_new_tokens=args.max_new_tokens
                # )
                
                #print(f"子问题回答: {output_sub}...")
                
                sub_total_count += 1
                is_sub_correct = sub_qa["ans"].lower() in output_sub.lower()
                print(f"子问题答案是否正确: {is_sub_correct}, 正确答案: {sub_qa['ans']}")
                if is_sub_correct:
                    sub_correct_count += 1
                
                result_sub = {
                    "id": f"{video_name}_{sub_qa_key}",
                    "type": "1",
                    "src_dataset": line["src_dataset"],
                    "video_name": video_name,
                    "prompt": qs_sub,
                    "pred_response": output_sub,
                    "ans": sub_qa["ans"],
                    "aspect": line["aspect"],
                    "is_correct": is_sub_correct
                }
                
                ans_file.write(json.dumps(result_sub) + "\n")
                ans_file.flush()
        
        # 处理字幕生成
        if "caption" in line:
            print("开始生成字幕...")
            caption_prompt = "Generate a comprehensive and concise description of the entire video, summarizing its key activities and events without focusing on individual frames."
            message_caption = [{'role': 'user', 'content': caption_prompt}]
            conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 16}},
                            {"type": "text", "text":caption_prompt},
                        ]
                    },
                ]
            inputs = processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            output_ids = model.generate(**inputs, max_new_tokens=1024)
            caption= processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            # caption = mm_infer(
            #     processed_video, 
            #     message_caption, 
            #     model, 
            #     tokenizer, 
            #     modal='video',
            #     do_sample=True if args.temperature > 0 else False,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     max_new_tokens=args.max_new_tokens
            # )
            
            caption = caption.replace("\n", " ").strip()
            caption = remove_duplicates_from_caption(caption)
            
            #print(f"生成的字幕: {caption[:50]}...")
            
            # 保存字幕结果
            result_caption = {
                "id": video_name,
                "type": "2",
                "src_dataset": line["src_dataset"],
                "video_name": video_name,
                "prompt": caption_prompt,
                "pred_response": caption,
                "ans": line["caption"],
                "aspect": line["aspect"]
            }
            
            ans_file.write(json.dumps(result_caption) + "\n")
            ans_file.flush()
            # 指定目标文件路径
            # file_path = "/root/autodl-tmp/VideoLLaMA3/output/test.jsonl"
            
            # 将 result_data 以 JSON 格式追加写入 .jsonl 文件
            # with open(file_path, 'a', encoding='utf-8') as f:
                #f.write(json.dumps(result_caption, ensure_ascii=False) + '\n')
            # caption_file.write(caption + "\n")
            # caption_answer_file.write(line["caption"] + "\n")
            #print("字幕生成完成")
    
    # 打印统计信息
    if ori_total_count > 0:
        print(f"原始问题正确率: {ori_correct_count}/{ori_total_count} = {ori_correct_count/ori_total_count*100:.2f}%")
    
    if sub_total_count > 0:
        print(f"子问题正确率: {sub_correct_count}/{sub_total_count} = {sub_correct_count/sub_total_count*100:.2f}%")
    
    # 关闭文件
    ans_file.close()
    # caption_file.close()
    # caption_answer_file.close()
    print(f"输出文件已关闭: {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="DAMO-NLP-SG/VideoLLaMA3-7B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video_aspect", type=str, default="reason_long")
    parser.add_argument("--video-folder", type=str, default="/root/autodl-tmp/VideoLLaMA3/data")
    parser.add_argument("--question-file", type=str, default="/root/autodl-tmp/VideoLLaMA3/std_answer/reason_long_result.jsonl")
    parser.add_argument("--answers-file", type=str, default="/root/autodl-tmp/VideoLLaMA3/results/videollama3/reason_long_result.jsonl")
    parser.add_argument("--for_get_frames_num", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--load_8bit", action="store_true", help="加载8-bit量化模型")
    parser.add_argument("--load_4bit", action="store_true", help="加载4-bit量化模型")
    parser.add_argument("--with-cot", type=bool, default=True)
    parser.add_argument("--remove-ans", type=bool, default=False)
    parser.add_argument("--with-partial-cot", type=bool, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_visual_tokens", type=int, default=12634)
    args = parser.parse_args()
    
    print(f"===== 启动 VideoLlama3 评估脚本 =====")
    print(f"参数配置:")
    print(f"  模型路径: {args.model_path}")
    print(f"  视频文件夹: {args.video_folder}")
    print(f"  问题文件: {args.question_file}")
    print(f"  输出文件: {args.answers_file}")
    
    try:
        eval_model(args)
        print(f"===== 评估完成 =====")
    except Exception as e:
        print(f"程序异常终止: {str(e)}")
        traceback.print_exc()
        exit(1)
