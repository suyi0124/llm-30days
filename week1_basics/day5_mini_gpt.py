# ============================================================
# Day 5 · 手写一个 mini GPT（字符级，CPU 可跑）
# 结构和大模型完全一致：Token嵌入 + 位置嵌入 → 因果自注意力 → 输出层
# 规模缩到极小：n_embd=32, block_size=32, 约 5 万参数
# 运行：python day5_mini_gpt.py       （CPU 约 1-3 分钟）
# ============================================================
#
# 看懂这份代码的顺序（重要，别从上往下硬读）：
#   第 0 步 数据：文字 ↔ 数字
#   第 1 步 单个注意力头（今天的核心，就是昨天的公式 + 掩码）
#   第 2 步 拼成完整模型（嵌入 + 注意力 + 输出层）
#   第 3 步 训练（还是那五行循环）
#   第 4 步 生成文本（模型自己往下写）

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)   # 固定种子，结果可复现


# ============================================================
# 第 0 步 · 数据：把文字变成数字
# ============================================================
# 为了不依赖下载，这里用一小段内置文本当语料。
# 真实项目用的是几十 GB 的网页文本，机制一模一样。
text = """
to be or not to be that is the question
whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune
or to take arms against a sea of troubles
and by opposing end them to die to sleep
no more and by a sleep to say we end
the heartache and the thousand natural shocks
that flesh is heir to tis a consummation
devoutly to be wished to die to sleep
to sleep perchance to dream ay there is the rub
for in that sleep of death what dreams may come
when we have shuffled off this mortal coil
must give us pause there is the respect
that makes calamity of so long life
"""

# 词表：这段文本里出现过的所有字符（去重排序）
chars = sorted(set(text))
vocab_size = len(chars)
print(f"词表大小 vocab_size = {vocab_size}（每个字符是一个 token）")

# 两个字典：字符→编号，编号→字符
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]          # "to be" → [7, 3, ...]
decode = lambda ids: "".join(itos[i] for i in ids)  # [7, 3] → "to"

data = torch.tensor(encode(text), dtype=torch.long)
print(f"整个语料共 {len(data)} 个 token")

# 划出 90% 训练、10% 验证（验证集用来判断有没有"背下来"）
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 32    # 一次最多看 32 个字符（上下文长度，大模型是 4k-128k）
batch_size = 16    # 一次并行训练 16 段

def get_batch(split="train"):
    """随机取一批训练数据。
    x: (16, 32)  输入的 32 个字符
    y: (16, 32)  目标 —— 每个位置要预测的"下一个字符"
    注意 y 就是 x 整体左移一位：看前 32 个，猜后 32 个。
    """
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i: i + block_size] for i in ix])
    y = torch.stack([d[i + 1: i + block_size + 1] for i in ix])
    return x, y

xb, yb = get_batch()
print(f"一个 batch：x{xb.shape}  y{yb.shape}")
print(f"检查：x 的第一个样本开头 = {decode(xb[0][:10].tolist())!r}")
print(f"      y 的第一个样本开头 = {decode(yb[0][:10].tolist())!r}   ← 整体左移一位")
print()


# ============================================================
# 第 1 步 · 单个注意力头（今天的核心）
# ============================================================
class Head(nn.Module):
    """一个因果自注意力头。就是昨天的 softmax(QKᵀ/√d_k)V，只多了一个掩码。"""

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        # 三个线性层，把输入分别投影成 Q、K、V
        # bias=False 是 GPT 的惯例（LayerNorm 会处理偏移）
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # 因果掩码：下三角矩阵，位置 i 只能看到 <= i 的位置
        # register_buffer 表示"这是个常量，不算可训练参数"，但会跟着模型存盘/搬设备
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape            # 批大小, 序列长度, 每token维度

        k = self.key(x)              # (B, T, head_size)
        q = self.query(x)            # (B, T, head_size)

        # 打分：每个位置跟每个位置的相关度 → (B, T, T)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        # 除以 √head_size，昨天讲过：防止 softmax 太极端

        # 掩码：上三角（未来位置）填 -inf，softmax 后变成 0 → 不能偷看未来
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)   # 每一行和为 1，就是昨天那张百分比表

        v = self.value(x)              # (B, T, head_size)
        out = wei @ v                  # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out


# ============================================================
# 第 2 步 · 拼成完整模型
# ============================================================
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=32, block_size=32):
        super().__init__()
        self.block_size = block_size

        # ① 词嵌入：每个 token 编号 → 一个 32 维向量
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # ② 位置嵌入：每个位置（0~31）→ 一个 32 维向量
        #    为什么需要？注意力本身对顺序无感（打乱顺序结果一样），必须额外告诉它"谁在前面"
        self.position_embedding = nn.Embedding(block_size, n_embd)

        # ③ 注意力
        self.sa = Head(n_embd, n_embd, block_size)

        # ④ 输出层：32 维 → 词表大小，每个字符得一分
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)                        # (B, T, 32)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, 32)
        x = tok_emb + pos_emb                                      # 广播相加 → (B, T, 32)

        x = self.sa(x)                                             # (B, T, 32)
        logits = self.lm_head(x)                                   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # 交叉熵要求 (样本数, 类别数)：把 B、T 两维拍平，再算每个位置的分类损失
            B_, T_, V_ = logits.shape
            loss = F.cross_entropy(logits.view(B_ * T_, V_), targets.view(B_ * T_))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """自回归生成：每次预测下一个字符，接到末尾，再预测下一个。"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]        # 只保留最后 32 个（模型只看得了这么长）
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                   # 只关心"最后一个位置"的预测
            probs = F.softmax(logits, dim=-1)           # 变成概率
            idx_next = torch.multinomial(probs, num_samples=1)  # 按概率抽一个字符
            idx = torch.cat((idx, idx_next), dim=1)     # 接上去
        return idx


model = MiniGPT(vocab_size, n_embd=32, block_size=block_size)
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
print()


# ============================================================
# 第 3 步 · 训练（还是那五行）
# ============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(1500):
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)     # ① 前向 + 算损失（封装在一起了）
    optimizer.zero_grad(set_to_none=True)   # ② 清零
    loss.backward()                  # ③ 反向
    optimizer.step()                 # ④ 更新

    if step % 300 == 0:
        # 顺便看看验证集 loss：训练 loss 降但验证 loss 升 = 过拟合（背答案）
        xv, yv = get_batch("val")
        _, val_loss = model(xv, yv)
        print(f"step {step:4d} | 训练loss={loss.item():.4f} | 验证loss={val_loss.item():.4f}")

# 预期：训练 loss 从 4.2 左右（≈ln(词表大小)，纯瞎猜）降到 1.0 以下
print()


# ============================================================
# 第 4 步 · 生成文本
# ============================================================
start = torch.zeros((1, 1), dtype=torch.long)   # 从编号 0 的字符开始
generated = model.generate(start, max_new_tokens=300)
print("生成结果：")
print(decode(generated[0].tolist()))


# ============================================================
# 自测（跑通后不看代码回答）
# ============================================================
# 1. 为什么 x 和 y 是整体错一位的关系？
# 2. 位置嵌入解决什么问题？去掉它会怎样？
# 3. 掩码那行如果不写，模型会"作弊"成什么样？
# 4. 生成时为什么每次只取 logits[:, -1, :]？
# 5. 训练 loss 0.3、验证 loss 2.0，说明什么？
#
# 下一步（Day 6）：加多头注意力、残差连接、LayerNorm、FFN，
# 把这份代码升级到和 GPT-2 同构，然后对比效果差异。
