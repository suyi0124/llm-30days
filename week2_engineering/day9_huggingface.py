import os
# 镜像设置必须放在 import transformers 之前！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F

from transformers import pipeline

print("=" * 60)
print("Part 0 · pipeline 一行代码")

classifier = pipeline("sentiment-analysis")

results = classifier([
    "This movie is absolutely wonderful, I loved every minute of it.",
    "What a waste of two hours. The plot was boring and the acting was terrible.",
])

for r in results:
    print(f"  {r['label']:10s} 置信度 {r['score']:.4f}")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ① 加载 tokenizer 和模型（第一次运行会自动下载，约 270MB）
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()   # 推理模式（Day 3 学过：关掉 dropout 等训练行为）

text = "I really enjoyed this film!"

# ② 预处理：文字 → token ID（昨天学的内容在这里上岗！）
inputs = tokenizer(text,return_tensors="pt")  # pt = 返回 PyTorch 张量
print(f"  tokenizer 输出的键: {list(inputs.keys())}")
print(f"  input_ids 形状: {inputs['input_ids'].shape}")   # (1, 序列长度)
# 注意有 'attention_mask' 键：1 表示真实 token，0 表示补齐的填充（padding）
# batch 里句子长短不一时靠它告诉模型"哪些位置不算"

# 注意有 'attention_mask' 键：1 表示真实 token，0 表示补齐的填充（padding）
# batch 里句子长短不一时靠它告诉模型"哪些位置不算"

# ③ 前向传播：模型给出每个类别的"原始分数"（logits）
with torch.no_grad():
    outputs = model(**inputs)   # ** 把字典拆成关键字参数传进去

logits = outputs.logits
print(f"  logits: {logits}")            # 形状 (1, 2)：两个类别的原始分数
print(f"  logits 形状: {logits.shape}")

# ④ softmax 转成概率（Day 4 学的！）
probs = F.softmax(logits, dim=-1)
print(f"  概率: {probs}")               # 两类概率，和为 1

# ⑤ argmax 取出答案（Day 3 学的！）
pred_id = probs.argmax(dim=-1).item()
pred_label = model.config.id2label[pred_id]
pred_score = probs.max().item()
print(f"  预测: {pred_label} ({pred_score:.4f})")

# ✅ 检查点：这里的结果应该和 Part 0 的 pipeline 完全一致
# —— 证明 pipeline 没有任何魔法，只是把这 5 步包了起来


# Part 2 · datasets：加载真实数据集
# ============================================================
# IMDb：5 万条电影评论，标注了好评/差评，文本分类最经典的数据集。
# 第 2 周 Day 10 微调 BERT 就用它，今天先认识它。

print("=" * 60)
print("Part 2 · datasets 加载 IMDb")

from datasets import load_dataset

# 只加载训练集（streaming 或小配置可以省流量，这里直接下完整训练集约 80MB）
dataset = load_dataset("stanfordnlp/imdb", split="train")

print(f"  数据集大小: {len(dataset)} 条")
print(f"  字段: {dataset.column_names}")        # ['text', 'label']
print(f"  第 0 条标签: {dataset[0]['label']}")   # 0=差评 1=好评
print(f"  第 0 条文本开头: {dataset[0]['text'][:80]}...")

# 数据集支持切片、过滤、map——和 Python 列表很像但更快
short_reviews = dataset.filter(lambda x: len(x["text"]) < 500)
print(f"  短评数量: {len(short_reviews)} 条")

# 标签分布（ Classics：确认两类是否均衡）
from collections import Counter
labels = [dataset[i]["label"] for i in range(1000)]
print(f"  前 1000 条的标签分布: {Counter(labels)}")

print("=" * 60)
print("Part 3 · 用模型给 8 条影评打分")

samples = short_reviews.shuffle(seed=42).select(range(8))
correct = 0

for item in samples:
    text = item["text"]
    true_label = item["label"]

    # 复用 Part 1 的 5 步（③④⑤合写）
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = logits.argmax(dim=-1).item()

    mark = "✓" if pred == true_label else "✗"
    correct += (pred == true_label)
    print(f"  [{mark}] 真实={true_label} 预测={pred} | {text[:50]}...")

print(f"  准确率: {correct}/8 = {correct/8:.0%}")
# 这个模型在 SST-2 上训练过，迁移到 IMDb 一般能对 6-8 条——
# 这就是"预训练+微调"范式的威力：没见过 IMDb 也能直接用

print("=" * 60)
print("全部跑通！Day 9 的核心任务完成。")
print("接下来做文件末尾的 3 个自测实验。")
