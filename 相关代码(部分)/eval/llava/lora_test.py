from datasets import Dataset
import cv2
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

tokenizer = AutoTokenizer.from_pretrained("lmms-lab/llava-onevision-qwen2-7b-ov")

def read_jsonl_file(jsonl_file_path):
    data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_data = json.loads(line.strip())
            data.append(json_data)
    return data

jsonl_data = read_jsonl_file('autodl-tmp/errors.jsonl')

def extract_video_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

integrated_dataset = []
for json_data in jsonl_data:
    video_name = json_data['video_name']
    video_path = f'autodl-tmp/data/{video_name}' 
    frames = extract_video_frames(video_path)
    json_data['frames'] = frames
    integrated_dataset.append(json_data)
    
ds = Dataset.from_dict({
    'video_name': [data['video_name'] for data in integrated_dataset],
    'aspect': [data['aspect'] for data in integrated_dataset],
   'src_dataset': [data['src_dataset'] for data in integrated_dataset],
    'original_qa': [data['original_qa'] for data in integrated_dataset],
   'sub_qas': [data['sub_qas'] for data in integrated_dataset],
    'caption': [data['caption'] for data in integrated_dataset],
    'frames': [data['frames'] for data in integrated_dataset]
})

tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]["aspect"])))

model = AutoModelForCausalLM.from_pretrained("lmms-lab/llava-onevision-qwen2-7b-ov")

for name, parameter in model.named_parameters():
    print(name)
model = get_peft_model(model, config)
for name, parameter in model.named_parameters():
    print(name)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="./chatbot",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()
