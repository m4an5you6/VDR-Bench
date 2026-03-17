import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

from bert_score import score

# 读取参考文本和模型预测文本
with open("model_predict.txt") as f:
    refs = [line.strip() for line in f]

with open("referrence.txt") as f:
    cands = [line.strip() for line in f]



# 初始化 F1 分数总和
total_F1 = 0.0

# 逐行计算 F1 分数并累加
for i, (cand, ref) in enumerate(zip(cands, refs)):
    P, R, F1 = score([cand], [ref], lang='en', verbose=True)
    total_F1 += F1.mean()  # 累加 F1 分数
    print(f"Line {i+1} - F1 score with original reference: {F1.mean():.3f}")

# 计算平均 F1 分数
average_F1 = total_F1 / len(cands)
print(f"Average F1 score: {average_F1:.3f}")