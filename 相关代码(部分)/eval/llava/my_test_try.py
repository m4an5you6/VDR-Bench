from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
import json
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer,BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import  AutoTokenizer
import re

warnings.filterwarnings("ignore")
# 指定缓存目录
cache_dir = "/root/autodl-tmp/huancun"
# 加载OneVision模型
pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
}
#tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args)
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", cache_dir=cache_dir, **llava_model_args
)
model.eval()


# 提取视频帧的函数
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)

# 读取JSONL文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# 准备对话输入
def prepare_input(video_frames, question, choices):
    conv_template = "qwen_1_5"
    question=question+choices+"按问题选择你的选项，最后只输出你的选项，"
    question_final=f"{DEFAULT_IMAGE_TOKEN}\n{question}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question_final)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    # 检查 choices 的类型
    if isinstance(choices, dict):
        # 如果是字典，按原来的方式处理
        choices_str = "\n".join([f"选项{key}: {value}" for key, value in choices.items()])
    elif isinstance(choices, str):
        # 如果是字符串，直接使用它
        choices_str = choices
    else:
        # 如果不是字典也不是字符串，抛出异常或做其他处理
        raise ValueError("choices should be either a dictionary or a string")

    """ prompt_question += f"\n{choices_str}"
    prompt_question += f"\n按问题选择你的选项，最后只输出你的选项，只有A,B两个选项" """

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [frame.size for frame in video_frames]
    return input_ids, image_sizes


# 生成响应
def generate_answer(input_ids, image_tensors,image_sizes):
        # 生成响应
    with torch.no_grad():
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            modalities=["video"],
        )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    # 后处理生成结果，只保留选项（A 或 B）
    predicted_answer = text_outputs[0].strip()
    match = re.search(r'\b[AaBb]\b', predicted_answer)
    predicted_answer = match.group(0).upper()
    #print(f"Generated Answer: {predicted_answer}")
    return predicted_answer

# 生成响应
def generate_caption(input_ids, image_tensors,image_sizes):
        # 生成响应
    with torch.no_grad():
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            modalities=["video"],
        )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    # 后处理生成结果，只保留选项（A 或 B）
    predicted_answer = text_outputs[0].strip()
    #print(f"Generated Answer: {predicted_answer}")
    return predicted_answer

# 计算准确度
def compute_accuracy(predicted_answer, ground_truth_answer):
    return predicted_answer == ground_truth_answer

def calculate_cosine_similarity(caption1, caption2):
    # 使用TF-IDF向量化
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([caption1, caption2])

    # 计算余弦相似度
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    print(f"Cosine Similarity: {cosine_sim[0][0]}")
    return cosine_sim[0][0]
def main():
    jsonl_file_path = "pace.jsonl"
    video_dir_path = "data"

    # 读取JSONL文件
    data = read_jsonl(jsonl_file_path)

    total_qa_count = 0
    correct_qa_count = 0
    total_caption_similarity = 0.0
    caption_count = 0

    # 用来存储所有原始问题和子问题的比较结果
    comparisons = []
    # 遍历每个问题
    for item in data:
        video_name = item["video_name"]
        video_file_path = f"{video_dir_path}/{video_name}"

        # 加载视频帧
        video_frames = load_video(video_file_path, 16)
        print(video_frames.shape)  # (16, 1024, 576, 3)

        # 处理视频帧
        frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensors = [frames]

        original_qa = item["original_qa"]
        sub_qas = item["sub_qas"]
        ground_truth_caption = item["caption"]

        # 处理主问题
        question = original_qa["qs"]
        choices = original_qa["choice"]
        choices_str = ", ".join([f"{key}: {value}" for key, value in choices.items()])
        ground_truth_answer = original_qa["ans"]
        print(f"Question: {question}")
        print(f"Choices: {choices_str}")
        # 准备多模态模型的输入
        input_ids, image_sizes = prepare_input(video_frames, question, choices_str)

        # 生成响应
        predicted_answer = generate_answer(input_ids, image_tensors,image_sizes)

        # 记录比较结果
        comparisons.append({
            "question": question,
            "predicted_answer": predicted_answer,
            "ground_truth_answer": ground_truth_answer,
            "is_correct": compute_accuracy(predicted_answer, ground_truth_answer)
        })

        # 计算准确度
        correct_qa_count += compute_accuracy(predicted_answer, ground_truth_answer)
        total_qa_count += 1
        # 生成caption
        caption_input_ids, caption_image_sizes = prepare_input(video_frames, "These are frames from a video that I want to upload. Generate a detailed description that I can upload along with the video.", "")
        predicted_caption = generate_caption(caption_input_ids, image_tensors,caption_image_sizes)
        print(f"caption: {predicted_caption}")

        # 处理子问题
        for sub_qa in sub_qas.values():
            question = sub_qa["qs"]
            choices = sub_qa["choice"]
            choices_str = ", ".join([f"{key}: {value}" for key, value in choices.items()])
            ground_truth_answer = sub_qa["ans"]

            # 准备多模态模型的输入
            input_ids, image_sizes = prepare_input(video_frames, question, choices_str)

            # 生成响应
            predicted_answer = generate_answer(input_ids, image_tensors,image_sizes)

            # 记录比较结果
            comparisons.append({
                "question": question,
                "predicted_answer": predicted_answer,
                "ground_truth_answer": ground_truth_answer,
                "is_correct": compute_accuracy(predicted_answer, ground_truth_answer)
            })

            # 计算准确度
            correct_qa_count += compute_accuracy(predicted_answer, ground_truth_answer)
            total_qa_count += 1
            #print(f"Sub-Question: {question}")
            #print(f"Choices: {choices_str}")
        # 计算相似度并输出结果
        similarity = calculate_cosine_similarity(predicted_caption, ground_truth_caption)
        if similarity is not None:  # 检查相似度是否为None
            caption_count += 1
            total_caption_similarity += similarity
            print(f"Cosine Similarity: {similarity}")
        else:
            print("Similarity calculation returned None.")
        # 计算总体准确度和平均余弦相似度
    overall_qa_accuracy = correct_qa_count / total_qa_count
    overall_caption_accuracy=total_caption_similarity/caption_count

    # 统一展示结果
    #print(f"Overall QA Accuracy: {overall_qa_accuracy:.4f}")
    #print(f"Overall Caption Accuracy: {overall_caption_accuracy:.4f}")

    # 打印模型的输出与标准答案比较结果
    print("\nComparison of Model Answers with Ground Truth:")
    """ for comparison in comparisons:
        print(f"Question: {comparison['question']}")
        print(f"Predicted Answer: {comparison['predicted_answer']}")
        print(f"Ground Truth Answer: {comparison['ground_truth_answer']}")
        print(f"Correct: {'Yes' if comparison['is_correct'] else 'No'}")
        print("-" * 50) """

if __name__ == "__main__":
    main()