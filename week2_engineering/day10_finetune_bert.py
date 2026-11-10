# -*- coding: utf-8 -*-
"""
Day 10 · 微调 BERT 做情感分类（IMDb 影评）
==========================================
昨天你用现成模型"零样本"打分，准确率 75%。
今天用 3000 条标注数据"教"它，准确率能到 85%+。
这就是微调（fine-tuning）——让通用模型学会你的具体任务。

核心认知：训练循环和 Day 3 一模一样，还是那五行。
         变的只有三件事：数据变文本、模型换成 BERT、学习率变小 50 倍。

CPU 预计 6-10 分钟跑完（2 个 epoch）。有 GPU 会快 20 倍。

依赖：pip install datasets transformers
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 必须在 import transformers 前

import time
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------- 超参数 ----------------------------
MODEL_NAME = "distilbert-base-uncased"   # BERT 的轻量版：6 层，67M 参数
MAX_LEN    = 256    # 每条影评最多保留 256 个 token
BATCH_SIZE = 16
TRAIN_N    = 3000   # 只用 3000 条训练（全量 25000 条在 CPU 上要跑几小时）
TEST_N     = 800
EPOCHS     = 2
LR         = 2e-5   # ⚠️ 关键：微调学习率要小！Day 3 用的是 1e-3，这里小 50 倍

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if device.type == "cpu":
    print("提示：CPU 训练约 6-10 分钟，先去泡杯茶，别以为卡死了\n")


# ============================================================
# Part 0 · 准备数据（只取一小部分，够演示就行）
# ============================================================

print("=" * 60)
print("Part 0 · 加载数据")

# split 支持切片语法：train[:3000] = 训练集前 3000 条


# 改成：
train_ds = load_dataset("stanfordnlp/imdb", split=f"train[:{TRAIN_N}]")
test_ds  = load_dataset("stanfordnlp/imdb", split=f"test[:{TEST_N}]")


print(f"  训练集: {len(train_ds)} 条 | 测试集: {len(test_ds)} 条")
print(f"  第 0 条: label={train_ds[0]['label']}  text={train_ds[0]['text'][:60]}...")


# ============================================================
# Part 1 · 预处理：把 3000 条文本一次性全部变成 token
# ============================================================
# 昨天是一条一条 tokenizer(text)，今天用 dataset.map 批量处理。
# 关键参数 padding="max_length"：把所有句子补齐到同样长度 256，
# 这样它们才能叠成一个规整的张量 (16, 256) 一起送进模型。

print("=" * 60)
print("Part 1 · 批量分词")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_fn(batch):
    """batch 是一个字典：{'text': [3000条文本], 'label': [...]}"""
    return tokenizer(
        batch["text"],
        padding="max_length",   # 补到 max_length，短句后面用 [PAD] 填满
        truncation=True,        # 超过 256 的截断
        max_length=MAX_LEN,
    )


# batched=True：一次喂 1000 条进去，比逐条快得多
train_ds = train_ds.map(tokenize_fn, batched=True)
test_ds  = test_ds.map(tokenize_fn, batched=True)

# HuggingFace 模型要求标签字段名必须叫 "labels"（不是 "label"）
train_ds = train_ds.rename_column("label", "labels")
test_ds  = test_ds.rename_column("label", "labels")

# 只保留模型需要的三列，并转成 PyTorch 张量
cols = ["input_ids", "attention_mask", "labels"]
train_ds.set_format("torch", columns=cols)
test_ds.set_format("torch", columns=cols)

print(f"  分词后一条样本的字段: {train_ds[0].keys()}")
print(f"  input_ids 形状: {train_ds[0]['input_ids'].shape}")   # (256,)

# DataLoader 直接吃 HuggingFace 数据集，和 Day 3 用法完全一样
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32)


# ============================================================
# Part 2 · 加载模型（换个"头"）
# ============================================================
# AutoModelForSequenceClassification 会自动在 BERT 主体上加一个
# 分类头：768 维 → 2 类。这个头是随机初始化的，所以会看到
# "newly initialized" 警告——正常，它就是要被训练的部分。

print("=" * 60)
print("Part 2 · 加载模型")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,      # 二分类：差评 / 好评
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  总参数: {total_params:,} | 可训练: {trainable:,}")
# 微调 = 全部参数都参与更新（后面 Day 13 的 LoRA 只更新 1%）

# AdamW 是训练 Transformer 的标准优化器（比 Day 3 的 Adam 多了权重衰减）
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


# ============================================================
# Part 3 · 训练（还是那五行！）
# ============================================================

def evaluate(loader):
    """在测试集上算准确率 —— 和 Day 3 的评估完全一样"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


print("=" * 60)
print("Part 3 · 开始训练")

for epoch in range(EPOCHS):
    model.train()
    total_loss = correct = total = 0
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # ① 前向：传 labels 进去，模型内部自己算交叉熵损失
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss            # 已经是一个标量，不用再自己写 loss_fn

        # ② 反向 ③ 更新 ④ 清零（和 Day 2/3 一字不差）
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if step % 50 == 0:
            print(f"  epoch {epoch+1} | step {step:4d}/{len(train_loader)} "
                  f"| loss {loss.item():.4f} | 累计acc {correct/total:.3f}")

    train_acc = correct / total
    test_acc = evaluate(test_loader)
    print(f"  ✅ epoch {epoch+1} 完成 | 平均loss {total_loss/len(train_loader):.4f} "
          f"| 训练acc {train_acc:.4f} | 测试acc {test_acc:.4f} "
          f"| 用时 {time.time()-t0:.1f}s\n")


# ============================================================
# Part 4 · 保存 + 用自己训的模型预测
# ============================================================

print("=" * 60)
print("Part 4 · 保存并试用")

save_dir = "./day10_bert_imdb"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"  已保存到 {save_dir}/")

# 从保存的目录重新加载（验证保存成功）
loaded_model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(device)
loaded_model.eval()


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
    with torch.no_grad():
        logits = loaded_model(**inputs).logits
    prob = torch.softmax(logits, dim=-1)[0]
    label = "好评" if prob.argmax().item() == 1 else "差评"
    return label, prob.max().item()


# 试试自己的句子（模型只在影评上训过，其他领域可能不准——这是正常现象）
my_texts = [
    "This film is a masterpiece. The acting is superb and the story moved me deeply.",
    "Boring, predictable, and way too long. I fell asleep twice.",
    "It was okay, not great but not terrible either.",
]
for t in my_texts:
    label, score = predict(t)
    print(f"  [{label} {score:.3f}] {t[:60]}")

print("=" * 60)
print("Day 10 完成！对比一下：昨天零样本 75%，今天微调后 85%+")
print("接下来做文件末尾的三个实验。")
