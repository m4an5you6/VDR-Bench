import argparse
import os
import os.path as osp
import re
from io import BytesIO
import random
import string
import json
from tqdm import tqdm

import requests
import torch
from PIL import Image

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token, opencv_extract_frames
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

# 指定缓存目录
cache_dir = "/root/autodl-tmp/huancun"
# 用于存储已生成的 id
generated_ids = set()
def remove_duplicates_from_caption(caption):
    """
    从生成的 caption 中删除重复的句子或段落。
    :param caption: 生成的 caption 字符串
    :return: 去重后的 caption 字符串
    """
    # 将 caption 按句号分割为句子列表
    sentences = caption.split(". ")
    
    # 去重
    unique_sentences = []
    for sentence in sentences:
        # 去除句尾的句号
        sentence = sentence.strip(". ")
        # 如果句子不在唯一句子列表中，则添加
        if sentence not in unique_sentences:
            unique_sentences.append(sentence)
    
    # 将去重后的句子重新组合为字符串
    cleaned_caption = ". ".join(unique_sentences)
    return cleaned_caption
# 生成唯一的 8 位随机字符串
def generate_unique_id():
    while True:
        new_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        if new_id not in generated_ids:  # 检查是否已存在
            generated_ids.add(new_id)  # 添加到集合中
            return new_id
            
def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        print("downloading image from url", image_file)
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def eval_original_qa(line, unique_id, images, images_tensor, tokenizer, model, conv, ans_file, args):
    original_qa = line["original_qa"]
    qs = original_qa["qs"]
    choices_str = ', '.join([f"{key}: {value}" for key, value in original_qa["choice"].items()])
    qs = f"Question: {qs}, Choices: {choices_str}. Answer me using only the given options (A, B, C...)"

    # 处理问题中的图像占位符
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            cur_prompt = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            cur_prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if DEFAULT_IMAGE_TOKEN not in qs:
            if model.config.mm_use_im_start_end:
                cur_prompt = (image_token_se + "\n") * len(images) + qs
            else:
                cur_prompt = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs

    conv.append_message(conv.roles[0], cur_prompt)
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
            images=[images_tensor],
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

    # 保存 original_qa 结果
    result_data = {
        "id": unique_id,
        "type": "0",
        "src_dataset": line["src_dataset"],
        "video_name": line["video_name"],
        "prompt": qs,
        "pred_response": outputs,
        "ans": original_qa["ans"],
        "aspect": line["aspect"],
    }
    ans_file.write(json.dumps(result_data) + "\n")
    ans_file.flush()

    return outputs.upper() == original_qa["ans"].upper()

def eval_sub_qa(line, unique_id, images, images_tensor, tokenizer, model, conv, ans_file, args):
    sub_qas = line["sub_qas"]
    correct_sub_qa = 0
    total_sub_qa = 0

    for sub_qa_key, sub_qa in sub_qas.items():
        qs = sub_qa["qs"]
        choices_str = ', '.join([f"{key}: {value}" for key, value in sub_qa["choice"].items()])
        qs = f"Question: {qs}, Choices: {choices_str}. Answer me using only the given options (A, B, C...)"

        # 处理问题中的图像占位符
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                cur_prompt = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                cur_prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if DEFAULT_IMAGE_TOKEN not in qs:
                if model.config.mm_use_im_start_end:
                    cur_prompt = (image_token_se + "\n") * len(images) + qs
                else:
                    cur_prompt = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs

        conv.append_message(conv.roles[0], cur_prompt)
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
                images=[images_tensor],
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

        # 保存 sub_qa 结果
        result_data = {
            "id": unique_id,
            "type": "1",
            "src_dataset": line["src_dataset"],
            "video_name": line["video_name"],
            "prompt": qs,
            "pred_response": outputs,
            "ans": sub_qa["ans"],
            "aspect": line["aspect"],
        }
        ans_file.write(json.dumps(result_data) + "\n")
        ans_file.flush()

        total_sub_qa += 1
        if outputs.upper() == sub_qa["ans"].upper():
            correct_sub_qa += 1

    return total_sub_qa, correct_sub_qa

def eval_caption(line, unique_id, images, images_tensor, tokenizer, model, conv, ans_file, caption_file, args,caption_answer_file):
    #caption_prompt = "These is a video that I want to upload. Generate a detailed description that I can upload along with the video."
    caption_prompt ="Generate a comprehensive and concise description of the entire video, summarizing its key activities and events without focusing on individual frames."
    caption_prompt = "<image>\n" + caption_prompt
    images_tensor = torch.mean(images_tensor, dim=0, keepdim=True)
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
            images=[images_tensor],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    #print(outputs)
    outputs = outputs.strip()
    
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    outputs= remove_duplicates_from_caption(outputs)
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
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Data
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # 读取 JSONL 文件
    questions = []
    with open(os.path.expanduser(args.question_file), "r") as f:
        for eachline in f:
            questions.append(json.loads(eachline))

    # 用于计算准确度
    total_original_qa = 0
    correct_original_qa = 0
    total_sub_qa = 0
    correct_sub_qa = 0

    # 用于保存 caption
    caption_file = open("/root/autodl-tmp/caption_eval/bert_score-master/referrence.txt", "w")
    caption_answer_file = open("/root/autodl-tmp/caption_eval/bert_score-master/model_predict.txt", "w")

    for line in tqdm(questions, desc="VILA eval"):
        # 生成唯一的 id
        unique_id = generate_unique_id()
        src_dataset = line["src_dataset"]
        video_file = line["video_name"]
        #video_path = os.path.join(args.video_folder, src_dataset)
        #video_path = os.path.join(video_path, video_file)
        video_path = os.path.join(args.video_folder, video_file)

        # 提取视频帧
        images, num_frames = opencv_extract_frames(video_path, args.num_video_frames)

        # 处理图像
        images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
        #images_tensor = torch.mean(images_tensor, dim=0, keepdim=True)
        #print(f"Image tensor shape: {images_tensor.shape}")
        # 设置对话模板
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()

        # 第一部分：处理 original_qa
        total_original_qa += 1
        correct_original_qa += eval_original_qa(line, unique_id, images, images_tensor, tokenizer, model, conv, ans_file, args)
        # 重置对话模板
        conv = conv_templates[conv_mode].copy()
        # 第二部分：处理 sub_qa
        sub_qa_total, sub_qa_correct = eval_sub_qa(line, unique_id, images, images_tensor, tokenizer, model, conv, ans_file, args)
        total_sub_qa += sub_qa_total
        correct_sub_qa += sub_qa_correct
        # 重置对话模板
        conv = conv_templates[conv_mode].copy()
        # 第三部分：处理 caption
        eval_caption(line, unique_id, images, images_tensor, tokenizer, model, conv, ans_file, caption_file, args,caption_answer_file)

    # 关闭文件
    ans_file.close()
    caption_file.close()
    
    # 打印准确度
    print(f"Original QA Accuracy: {correct_original_qa / total_original_qa * 100:.2f}%")
    print(f"Sub QA Accuracy: {correct_sub_qa / total_sub_qa * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA1.5-7b")
    parser.add_argument("--model-path", type=str, default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
    #parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/Llama-3-VILA1.5-8B")
    #parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/qwen2-7b-longvila-256f")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="/root/autodl-tmp/question/unique_change_long.jsonl")
    #parser.add_argument("--answers-file", type=str, default="/root/autodl-tmp/results/VILA1.5-3b/answer.jsonl")
    parser.add_argument("--answers-file", type=str, default="/root/autodl-tmp/results/VILA1.5-7b/change_long.jsonl")
    #parser.add_argument("--answers-file", type=str, default="/root/autodl-tmp/results/longVILA/Prediction_medium.jsonl")
    parser.add_argument("--video-folder", type=str, default="/root/autodl-tmp/data/change/long/long")
    parser.add_argument("--num-video-frames", type=int, default=16)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1") # vicuna_v1
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()

    eval_model(args)
