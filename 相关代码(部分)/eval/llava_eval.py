import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import math
from decord import VideoReader, cpu
from transformers import AutoConfig
import cv2
import base64
from PIL import Image
import numpy as np
from typing import Dict, List
import re
import time

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llava.mm_utils import KeywordsStoppingCriteria
import warnings

import os


warnings.filterwarnings("ignore")
device = "cuda"


def remove_answers(sub_qas):
    for sub_qa in sub_qas.values():
        if "ans" in sub_qa:
            del sub_qa["ans"]
    return sub_qas


def load_video(video_path, args):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    if len(frame_idx) > args.for_get_frames_num:
        sample_fps = args.for_get_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames


def preprocess_qwen(sources, tokenizer: torch.nn.Module, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split("<image>")
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i, text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i < len(texts) - 1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i == IMAGE_TOKEN_INDEX for i in _input_id]) == num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    # input_ids = torch.tensor(input_ids, dtype=torch.bfloat16)
    # targets = torch.tensor(targets, dtype=torch.bfloat16)
    return input_ids


def format_question_and_choices(question, choices):
    """格式化问题和选项字符串"""
    choices_str = ", ".join([f"{key}: {value}" for key, value in choices.items()])
    return f"Question: {question}, Choices: {choices_str}."

def generate_cot(qs, model, tokenizer, video=None, image_sizes=None, args=None, sub_qas=None):
    # 构建 COT prompt
    if args.with_partial_cot:
        cot_prompt = f"{qs}Let's think step by step.Please help me generate sub-questions based on the above question and provide the corresponding answers for analysis of the above question later.Among these sub-questions, the phrase sub questions that I provided below must be included:{remove_answers(sub_qas) if args.remove_ans else sub_qas}"
    else:
        cot_prompt = f"{qs}Let's think step by step.Please help me generate sub-questions based on the above question and provide the corresponding answers for analysis of the above question later."
    # 预处理输入，确保格式正确
    # input_ids = preprocess_qwen([{'from': 'human', 'value': cot_prompt}, {'from': 'gpt', 'value': None}], tokenizer, has_image=True).cuda()
    input_ids = tokenizer_image_token(cot_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    # 调用模型生成 COT
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=video,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=512,
            use_cache=True,
            image_sizes=image_sizes,
            modalities=["video"],
        )

    # 解码生成的 COT
    cot_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    # 返回生成的 COT
    return cot_output

def remove_duplicates_from_caption(caption):
    # 这里可以添加一些逻辑来去除重复的内容
    # 例如，去除重复的句子或短语
    return caption

def eval_caption(line, unique_id, video, tokenizer, model, conv, ans_file, caption_file, args, caption_answer_file):
    caption_prompt = "Generate a comprehensive and concise description of the entire video, summarizing its key activities and events without focusing on individual frames."
    caption_prompt = "<image>\n" + caption_prompt
    video_tensor = torch.mean(video, dim=0, keepdim=True)
    conv.append_message(conv.roles[0], caption_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    # 设置停止条件
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # 生成回答
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=torch.ones(input_ids.shape, device=input_ids.device),
            images=[video_tensor],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    outputs = remove_duplicates_from_caption(outputs)

    # 保存 caption 结果
    result_data = {
        "id": unique_id,
        "type": "2",
        "src_dataset": line["src_dataset"],
        "video_name": line["video_name"],
        "prompt": caption_prompt,
        "pred_response": outputs,
        "ans": line["caption"],
        "aspect": line["aspect"],
    }
    caption_answer_file.write(line["caption"] + "\n")
    ans_file.write(json.dumps(result_data) + "\n")
    ans_file.flush()

    # 去除换行符并保存 caption
    cleaned_caption = outputs.replace("\n", " ").strip()
    caption_file.write(cleaned_caption + "\n")
    
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base)

    # Data
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    # 用于保存 caption
    caption_file = open("/root/autodl-tmp/caption eval/bert_score-master/referrence.txt", "w")
    caption_answer_file = open("/root/autodl-tmp/caption eval/bert_score-master/model_predict.txt", "w")
    questions = []
    with open(os.path.expanduser(args.question_file), "r") as f:
        for eachline in f:
            questions.append(json.loads(eachline))

    if hasattr(model.config, "num_video_frames") and model.config.num_video_frames is not None:
        num_video_frames = model.config.num_video_frames
    else:
        num_video_frames = 8

    if hasattr(model.config, "fps") and model.config.fps is not None:
        fps = model.config.fps
    else:
        fps = 0.0

    # 统计正确率
    ori_correct_count = 0  # original_qa 正确数量
    ori_total_count = 0    # original_qa 总数
    sub_correct_count = 0  # sub_qa 正确数量
    sub_total_count = 0    # sub_qa 总数

    for line in tqdm(questions):
        src_dataset = line["src_dataset"]
        video_file = line["video_name"]
        #video_path = os.path.join(args.video_folder, src_dataset)
        video_path = args.video_folder
        video_path = os.path.join(video_path, video_file)

        # Load and preprocess the video
        video = load_video(video_path, args)
        image_sizes = [frame.size for frame in video]
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()

        # Process original_qa
        original_qa = line["original_qa"]
        qs = format_question_and_choices(original_qa["qs"], original_qa["choice"])
        cur_prompt = "<image>\n" + qs + "\nFirst, analyze the video carefully. Then, select the most appropriate answer from the given options (A, B, C, ...)."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], cur_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        # Inference for original_qa
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=torch.ones(input_ids.shape, device=input_ids.device),
                images=video,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
                image_sizes=image_sizes,
                modalities=["video"],
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # 检查 original_qa 答案是否正确
        ori_total_count += 1
        if original_qa["ans"].lower() in outputs.lower():  # 忽略大小写进行比较
            ori_correct_count += 1

        # Save the result for original_qa
        result_data = {
            "id": line["video_name"],  # 使用 video_name 作为唯一 ID
            "type": "0",  # original_qa 的 type 为 0
            "src_dataset": line["src_dataset"],
            "video_name": line["video_name"],
            "prompt": qs,
            "pred_response": outputs,
            "ans": original_qa["ans"],
            "aspect": line["aspect"],
        }
        ans_file.write(json.dumps(result_data) + "\n")
        ans_file.flush()

        # Process sub_qas
        sub_qas = line["sub_qas"]
        for sub_qa_key, sub_qa in sub_qas.items():
            qs = format_question_and_choices(sub_qa["qs"], sub_qa["choice"])
            cur_prompt = "<image>\n" + qs + "\nFirst, analyze the video carefully. Then, select the most appropriate answer from the given options (A, B, C, ...)."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], cur_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

            # Inference for sub_qa
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=torch.ones(input_ids.shape, device=input_ids.device),
                    images=video,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True,
                    image_sizes=image_sizes,
                    modalities=["video"],
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            # 检查 sub_qa 答案是否正确
            sub_total_count += 1
            if sub_qa["ans"].lower() in outputs.lower():  # 忽略大小写进行比较
                sub_correct_count += 1

            # Save the result for sub_qa
            result_data = {
                "id": line["video_name"],  # 使用 video_name 作为唯一 ID
                "type": "1",  # sub_qa 的 type 为 1
                "src_dataset": line["src_dataset"],
                "video_name": line["video_name"],
                "prompt": qs,
                "pred_response": outputs,
                "ans": sub_qa["ans"],
                "aspect": line["aspect"],
            }
            ans_file.write(json.dumps(result_data) + "\n")
            ans_file.flush()

        # Process caption
        if "caption" in line:
            eval_caption(line, line["video_name"], video, tokenizer, model, conv, ans_file, caption_file, args, caption_answer_file)

    # 计算并打印正确率
    ori_accuracy = ori_correct_count / ori_total_count if ori_total_count > 0 else 0
    sub_accuracy = sub_correct_count / sub_total_count if sub_total_count > 0 else 0
    print(f"Original QA: Total questions: {ori_total_count}, Correct answers: {ori_correct_count}, Accuracy: {ori_accuracy:.4f}")
    print(f"Sub QA: Total questions: {sub_total_count}, Correct answers: {sub_correct_count}, Accuracy: {sub_accuracy:.4f}")

    ans_file.close()
    caption_file.close()
    caption_answer_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="lmms-lab/llava-onevision-qwen2-7b-ov")
    # lmms-lab/llava-onevision-qwen2-0.5b-ov | /mnt/userData/zhouqiji/gyf_work/llava-onevision-qwen2-72b-ov-chat
    parser.add_argument("--model-base", type=str, default=None)
    #parser.add_argument("--video-folder", type=str, default="/root/autodl-tmp/data/sequence/medium/medium")
    parser.add_argument("--video-folder", type=str, default="/root/autodl-tmp/data/sequence/long/long")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--question-file", type=str, default="/root/autodl-tmp/sequence_long_result.jsonl") # partial_cot_split_150.jsonl | split_150.jsonl
    parser.add_argument("--answers-file", type=str, default="/root/autodl-tmp/results/llava-onevision/sequence_long_result.jsonl")
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--for_get_frames_num", type=int, default=16)
    parser.add_argument("--with-cot", type=bool, default=True)
    parser.add_argument("--remove-ans", type=bool, default=False)
    parser.add_argument("--with-partial-cot", type=bool, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()

    eval_model(args)
