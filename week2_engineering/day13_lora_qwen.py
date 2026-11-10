# -*- coding: utf-8 -*-
"""
Day 13 · LoRA 微调 Qwen2.5-0.5B
==============================
Day 10 你微调 BERT：全部 6700 万参数都更新，存出来的模型 270MB。
今天用 LoRA：只训练 0.3% 的参数，存出来的"补丁"只有几 MB。

一句话理解 LoRA：
    冻结原来的权重不动，在旁边挂两个小矩阵（A 和 B），
    只训练这两个小矩阵。d×d 的大矩阵 → r×d + d×r，参数量砍到 1/50。

为什么能这么干？因为微调需要的改动本来就是"小幅度调整"，
用低秩（rank 很小）的矩阵就足够表达。

CPU 预计 10-25 分钟（3 个 epoch，24 条数据）。
如果太慢：EPOCHS 改 1，MAX_LEN 改 128。

依赖：pip install peft accelerate transformers datasets
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"   # 5 亿参数，CPU 能跑
MAX_LEN    = 256
BATCH_SIZE = 2
EPOCHS     = 3
LR         = 2e-4    # LoRA 的学习率比全量微调大（因为只训小矩阵，不怕破坏原权重）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

DATA = [
    ("我的快递三天了还没到，太慢了", "非常抱歉让您久等了。我已为您查询物流进度，会尽快催促配送，请您耐心等待。"),
    ("你们这个产品质量太差了", "非常抱歉给您带来不好的体验。请您提供订单号，我们会立即为您安排退换处理。"),
    ("客服怎么一直不回我", "非常抱歉让您久等。我们的客服会在工作时间内尽快回复您，感谢您的理解。"),
    ("我要退款，快点", "非常理解您的心情。退款申请我会立即为您提交，通常在 1-3 个工作日内到账。"),
    ("这个价格太贵了", "感谢您的反馈。我们近期有优惠活动，您可以关注店铺公告，希望能为您带来更好的价格体验。"),
    ("东西收到了但是坏的", "非常抱歉给您添麻烦了。请拍照发给我们，我们会立即为您补发或退款。"),
    ("能不能便宜点", "感谢您的关注。目前我们有一个满减活动，您可以看看是否合适，我也可以帮您留意后续优惠。"),
    ("我要投诉你们", "非常抱歉让您不满意。请您详细说明情况，我会认真记录并向上反馈，尽快给您答复。"),
    ("什么时候发货", "您的订单预计在 24 小时内发出。发货后物流单号会同步更新，请您留意通知。"),
    ("发票怎么开", "您可以在订单详情页申请电子发票，填写抬头后系统会自动开具并发送到您的邮箱。"),
    ("买错了想换一个", "没问题。在未拆封的情况下支持 7 天无理由换货，我这就为您发起换货申请。"),
    ("会员有什么权益", "会员可享受专属折扣、优先发货和专属客服服务，具体权益可在会员中心查看。"),
    ("这个和另一款有什么区别", "两款产品在材质和容量上有所不同，我为您简单对比一下，您可以根据使用场景选择。"),
    ("我不小心下错单了", "别担心。如果订单还未发货，我可以帮您取消并重新下单，请您确认一下订单号。"),
    ("物流信息一直不更新", "抱歉让您担心了。物流信息有时会延迟同步，我已为您提交查询，稍后给您反馈。"),
    ("你们的东西是正品吗", "我们所有商品均为官方正品，支持专柜验货，请您放心购买。"),
    ("能不能加急发货", "我理解您的着急。已为您备注加急，仓库会优先处理，但具体时效仍以物流为准。"),
    ("退款什么时候到账", "退款一般 1-3 个工作日原路返回，具体以银行处理时间为准，请您留意账户变动。"),
    ("客服电话多少", "我们的客服热线是 400-000-0000，服务时间为每天 9:00-21:00，随时为您服务。"),
    ("包装破损了", "非常抱歉。请您保留外包装并拍照给我们，我们会为您补发或补偿处理。"),
    ("可以开专票吗", "可以的。请您提供公司名称和税号，我们会在 3 个工作日内为您开具增值税专用发票。"),
    ("我想取消订单", "好的。如订单未发货我可以立即为您取消，款项会原路退回，请您确认。"),
    ("东西比图片上小", "抱歉与您的预期有差异。商品页面有详细尺寸说明，如不合适我们可以为您办理退换。"),
    ("还有货吗", "这款目前有现货，下单后 24 小时内发出。如遇缺货我会第一时间通知您。"),
]
print(f"训练数据: {len(DATA)} 条")

# ============================================================
# Part 1 · 加载模型 + 套上 LoRA
# ============================================================

print("=" * 60)
print("Part 1 · 加载 Qwen2.5-0.5B（首次下载约 1GB）")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Qwen 的 pad_token 默认是 None，训练时补 padding 会报错，指定成 eos
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.to(device)

lora_config = LoraConfig(
    r=8,                      # 秩：越小参数越少，8 是常用起点
    lora_alpha=16,            # 缩放系数，一般设为 2*r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 给注意力的四个投影加 LoRA
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 预期看到类似：trainable params: 1,081,344 || all params: 494,032,768 || trainable%: 0.22
# ← 只有 0.2% 的参数参与训练，这就是 LoRA 的全部魔法

# ============================================================
# Part 2 · 构造训练样本（含 labels 掩码）
# ============================================================
# 关键细节：只对"回答"部分算 loss，对"提问"部分不算。
# 做法是把提问部分对应的 label 设为 -100（PyTorch 会忽略 -100）。

def build_example(q, a):
    prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    answer = f"{a}<|im_end|>\n"

    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    input_ids = (p_ids + a_ids)[:MAX_LEN]
    labels    = ([-100] * len(p_ids) + a_ids)[:MAX_LEN]   # 关键：前半段屏蔽
    return {"input_ids": input_ids, "labels": labels}

def collate(batch):
    """把一个 batch 里的样本补齐到同样长度"""
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids, labels, masks = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        ids.append(b["input_ids"] + [tokenizer.pad_token_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        masks.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(masks),
    }


examples = [build_example(q, a) for q, a in DATA]
print(f"  一条样本长度: {len(examples[0]['input_ids'])} tokens")
print(f"  其中被屏蔽(labels=-100)的位置: {examples[0]['labels'].count(-100)} 个")


# Part 3 · 训练（还是那五行）
# ============================================================

print("=" * 60)
print("Part 3 · 开始训练")

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss, steps = 0.0, 0
    t0 = time.time()

    # 简单的打乱（不用 DataLoader，数据量太小）
    order = torch.randperm(len(examples)).tolist()

    for i in range(0, len(order), BATCH_SIZE):
        batch = collate([examples[j] for j in order[i:i + BATCH_SIZE]])
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)      # ① 前向：内部自动算 loss
        loss = outputs.loss           # ② 拿损失
        loss.backward()               # ③ 反向
        optimizer.step()              # ④ 更新（只更新 LoRA 的小矩阵）
        optimizer.zero_grad()         # ⑤ 清零

        total_loss += loss.item()
        steps += 1
        if steps % 12 == 0:
            print(f"  epoch {epoch+1} | step {steps}/{len(order)//BATCH_SIZE} "
                  f"| loss {loss.item():.4f}")

    print(f"  ✅ epoch {epoch+1}: 平均 loss {total_loss/steps:.4f} | 用时 {time.time()-t0:.0f}s")

# ============================================================
# Part 4 · 前后对比 + 保存 adapter
# ============================================================

def generate(question, max_new_tokens=60):
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False,                     # 贪心解码，结果稳定好对比
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


test_qs = ["我买的东西坏了怎么办", "能便宜一点吗"]

print("=" * 60)
print("Part 4 · 训练前后对比")

model.eval()
print("\n--- 训练前（原版 Qwen）---")
with model.disable_adapter():        # 临时关掉 LoRA，用原始权重
    for q in test_qs:
        print(f"  Q: {q}\n  A: {generate(q)}\n")

print("--- 训练后（加载 LoRA）---")
for q in test_qs:
    print(f"  Q: {q}\n  A: {generate(q)}\n")

# 保存：只有 adapter，几 MB
save_dir = "./day13_lora_adapter"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

size_mb = sum(os.path.getsize(os.path.join(save_dir, f))
              for f in os.listdir(save_dir)) / 1024 / 1024
print(f"LoRA adapter 已保存: {save_dir}  总大小 {size_mb:.1f} MB")
print(f"对比：完整模型约 1000 MB —— 这就是 LoRA 的第四个好处：便宜好分发")

print("=" * 60)
print("Day 13 完成！")
print("思考：为什么 r=8 就够了？如果 r=64 会更准吗？(提示：过拟合 & 收益递减)")