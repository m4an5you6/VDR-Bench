import argparse
import os
import os.path as osp
import re
from io import BytesIO

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
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token,opencv_extract_frames
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

# 指定缓存目录
cache_dir = "/root/autodl-tmp/huancun"
def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        print("downloading image from url", args.video_file)
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
    questions = []
    with open(os.path.expanduser(args.question_file),"r") as f:
        for eachline in f:
            questions.append(json.loads(eachline))
            
    for line in tqdm(questions,desc="VILA eval"):
        src_dataset = line["src_dataset"]
        video_file = line["video_name"]
        #video_path = os.path.join(args.video_folder, src_dataset)
        video_path = os.path.join(video_path, video_file)

        images, num_frames = opencv_extract_frames(video_path, args.num_video_frames)

        # qs = line["question"]
        # if line["question_type"] != 2:
        #     choices_str = ', '.join([f"{key}: {value}" for key, value in line["choices"].items()])
        #     qs = f"Please answer the following question based on the video content.Question: {qs}, Choices: {choices_str}"
        # else: qs = f"Please answer the following question based on the video content.Question: {qs}"
        qs = line["original_qa"]["qs"]
        choices_str = ', '.join([f"{key}: {value}" for key, value in line["original_qa"]["choice"].items()])
        qs = f"Question: {qs}, Choices: {choices_str}.Answer me using only the given options (A,B,C...)"
        
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                cur_prompt = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                cur_prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if DEFAULT_IMAGE_TOKEN not in qs:
                # print("no <image> tag found in input. Automatically append one at the beginning of text.")
                # do not repeatively append the prompt.
                if model.config.mm_use_im_start_end:
                    cur_prompt = (image_token_se + "\n") * len(images) + qs
                else:
                    cur_prompt = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs
        # print("input: ", cur_prompt)
    
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
    
        # if args.conv_mode is not None and conv_mode != args.conv_mode:
        #     print(
        #         "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
        #             conv_mode, args.conv_mode, args.conv_mode
        #         )
        #     )
        # else:
        #     args.conv_mode = conv_mode
    
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], cur_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
        images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        # print(images_tensor.shape)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask = torch.ones(input_ids.shape, device=input_ids.device),
                images=[
                    images_tensor,
                ],
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
        # print(outputs)

        # Save the result
        result_data = {
            "src_dataset": line["src_dataset"],
            "video_name": line["video_name"],
            "prompt": qs,
            "pred_response": outputs,
            "ans": line["original_qa"]["ans"],
            "aspect": line["aspect"],
        }
        """ if line["question_type"] == 2:
            result_data["serial number"] = line["serial number"]
         """
        ans_file.write(json.dumps(result_data) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA1.5-3b")
    #parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/Llama-3-VILA1.5-8B")
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/Llama-3-LongVILA-8B-128Frames")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="/root/autodl-tmp/question/Prediction_medium_result.jsonl")
    #parser.add_argument("--answers-file", type=str, default="/root/autodl-tmp/results/VILA1.5-3b/answer.jsonl")
    #parser.add_argument("--answers-file", type=str, default="/root/autodl-tmp/results/VILA1.5-8b/answer.jsonl")
    parser.add_argument("--answers-file", type=str, default="/root/autodl-tmp/results/longVILA/Prediction_medium.jsonl")
    parser.add_argument("--video-folder", type=str, default="/root/autodl-tmp/data/root/autodl-tmp/Prediction/medium")
    parser.add_argument("--num-video-frames", type=int, default=16)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1") # vicuna_v1
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()

    eval_model(args)
