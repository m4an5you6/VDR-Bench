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
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
import re
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader

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

# 创建 Accelerator 实例
accelerator = Accelerator()

# 加载模型并准备
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", cache_dir=cache_dir, **llava_model_args
)

# 使用 accelerator 来准备模型（这会自动处理多显卡支持）
model = accelerator.prepare(model)  # 使用 accelerator.prepare() 以便支持多显卡

# 模型设置为评估模式
model.eval()
# 数据集类
from torch.utils.data import Dataset, DataLoader

class VideoQADataSet(Dataset):
    def __init__(self, data, video_dir_path, tokenizer, image_processor, max_frames=16):
        self.data = data
        self.video_dir_path = video_dir_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_frames = max_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_name = item["video_name"]
        video_file_path = f"{self.video_dir_path}/{video_name}"
        video_frames = load_video(video_file_path, self.max_frames)
        frames = self.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half()

        original_qa = item["original_qa"]
        question = original_qa["qs"]
        choices = original_qa["choice"]
        choices_str = ", ".join([f"{key}: {value}" for key, value in choices.items()])
        ground_truth_answer = original_qa["ans"]

        image_sizes = [(frame.shape[1], frame.shape[0]) for frame in video_frames]
        input_ids, _ = prepare_input(video_frames, question, choices_str)

        return input_ids.squeeze(0), frames, image_sizes, ground_truth_answer

    def collate_fn(self, batch):
        input_ids_list, frames_list, image_sizes_list, ground_truth_answer_list = zip(*batch)

        # 对 input_ids 进行填充
        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        # 将 frames 和 image_sizes 转换为张量
        frames_tensor = torch.stack(frames_list, dim=0)
        image_sizes_tensor = torch.tensor(image_sizes_list)

        # 将 ground_truth_answer 转换为张量
        ground_truth_answer_tensor = torch.tensor([ord(answer) - ord('A') for answer in ground_truth_answer_list])

        return input_ids_padded, frames_tensor, image_sizes_tensor, ground_truth_answer_tensor


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
    question = question + choices + "按问题选择你的选项，最后只输出你的选项，"
    question_final = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
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

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    # 生成 image_sizes 列表，确保它是 (width, height) 的元组
    image_sizes = [(frame.shape[1], frame.shape[0]) for frame in video_frames]
    return input_ids, image_sizes

# 生成响应
def generate_answer(input_ids, image_tensors, image_sizes):
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
    return predicted_answer

# 计算准确度
def compute_accuracy(predicted_answer, ground_truth_answer):
    return predicted_answer == ground_truth_answer

def train_model(model, dataloader, epochs=5):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scaler = GradScaler()  # 用于混合精度训练的缩放器

    for epoch in range(epochs):
        total_loss = 0
        for input_ids, image_tensors, image_sizes, ground_truth_answer in dataloader:
            input_ids, image_tensors, image_sizes, ground_truth_answer = accelerator.prepare(input_ids, image_tensors, image_sizes, ground_truth_answer)

            with autocast():  # 自动混合精度
                outputs = model(input_ids, images=image_tensors, image_sizes=image_sizes)
                logits = outputs.logits

                # 将答案转换为 0 或 1
                loss = criterion(logits, ground_truth_answer)

            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# 评估模型
def evaluate_model(model, data, video_dir_path):
    model.eval()
    total_qa_count = 0
    correct_qa_count = 0
    all_predictions = []
    all_ground_truth = []

    with torch.no_grad():
        for item in data:
            video_name = item["video_name"]
            video_file_path = f"{video_dir_path}/{video_name}"
            video_frames = load_video(video_file_path, 16)
            frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half()
            image_tensors = [frames]

            original_qa = item["original_qa"]
            question = original_qa["qs"]
            choices = original_qa["choice"]
            choices_str = ", ".join([f"{key}: {value}" for key, value in choices.items()])
            ground_truth_answer = original_qa["ans"]

            image_sizes = [(frame.shape[1], frame.shape[0]) for frame in video_frames]

            input_ids, image_sizes = prepare_input(video_frames, question, choices_str)

            # 准备数据
            input_ids, image_tensors, image_sizes = accelerator.prepare(input_ids, image_tensors, image_sizes)

            # 生成响应
            predicted_answer = generate_answer(input_ids, image_tensors, image_sizes)

            all_predictions.append(predicted_answer)
            all_ground_truth.append(ground_truth_answer)
            total_qa_count += 1

    # 使用 accelerate 收集所有 GPU 的结果
    all_predictions = accelerator.gather(all_predictions)
    all_ground_truth = accelerator.gather(all_ground_truth)

    # 计算准确率
    correct_qa_count = sum(p == g for p, g in zip(all_predictions, all_ground_truth))
    overall_qa_accuracy = correct_qa_count / total_qa_count
    print(f"Overall QA Accuracy: {overall_qa_accuracy:.4f}")
    return overall_qa_accuracy

def main():
    jsonl_file_path = "errors.jsonl"
    video_dir_path = "data"

    # 读取JSONL文件
    data = read_jsonl(jsonl_file_path)  # 加载数据并将其赋值给 `data`
    # 创建数据集和 DataLoader
    dataset = VideoQADataSet(data, video_dir_path, tokenizer, image_processor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)  # 调整批次大小

    # 训练模型
    train_model(model, dataloader, epochs=5)

    # 评估模型
    evaluate_model(model, data, video_dir_path)

    # 保存微调后的模型到新的文件
    torch.save(model.state_dict(), "llava_finetuned.pth")

if __name__ == "__main__":
    main()
