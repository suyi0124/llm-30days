# -*- coding: utf-8 -*-
"""
Day 2 完整参考实现：autograd → 手写训练循环 → nn.Module 版本
直接运行即可：python day2_full.py
建议学习顺序：先只看 Part 2 跑通，再对照 Part 3 理解两者的对应关系
"""
import torch
import torch.nn as nn

torch.manual_seed(42)          # 固定随机种子：保证每次运行结果一致，方便对比和调试
print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("=" * 70)


# ============================================================
# Part 1: autograd 热身 —— 理解"梯度"到底是什么
# ============================================================
print("\n【Part 1】autograd 基础\n")

# --- 1.1 最小例子：y = x²，在 x=3 处求导 ---
x = torch.tensor(3.0, requires_grad=True)
# requires_grad=True 是开关：打开后，PyTorch 会记录所有涉及 x 的运算，形成"计算图"
# 只有需要被优化的量（模型参数）才设 True；普通数据（输入 x、标签 y）不用设

y = x ** 2                      # 前向计算：此时计算图里记录了 "x 做了平方运算"
y.backward()                    # 反向传播：沿着计算图倒着走，自动算出 dy/dx
# 数学上 dy/dx = 2x = 6，结果被存进 x.grad
print(f"1.1  y=x², x=3 → x.grad={x.grad.item()}  (期望 6.0)")

# --- 1.2 线性函数：y = 3x + 2，梯度应恒为 3 ---
x = torch.tensor(1.0, requires_grad=True)
y = 3 * x + 2
y.backward()
print(f"1.2  y=3x+2 → x.grad={x.grad.item()}  (期望 3.0，斜率)")

# --- 1.3 链式法则：y = (x²+1)³，先手算再验证 ---
# 手算：dy/dx = 3(x²+1)² · 2x，x=1 时 = 3·4·2 = 24
x = torch.tensor(1.0, requires_grad=True)
y = (x ** 2 + 1) ** 3
y.backward()
print(f"1.3  y=(x²+1)³, x=1 → x.grad={x.grad.item()}  (手算 24.0)")

# --- 1.4 多级中间变量：链式法则自动串联 ---
# c = b², b = 3a → dc/da = dc/db · db/da = 2b · 3 = 2·6·3 = 36
a = torch.tensor(2.0, requires_grad=True)
b = a * 3
c = b ** 2
c.backward()
print(f"1.4  a=2, b=3a, c=b² → a.grad={a.grad.item()}  (期望 36.0)")
# 这一步的启示：无论模型多深（GPT 有几十上百层），autograd 都能自动串起来求导
# 这就是"反向传播"能在深度学习里可行的根本

# --- 1.5 大坑：梯度会累加，不清零就出错 ---
x = torch.tensor(2.0, requires_grad=True)
y1 = x ** 2
y1.backward()                   # 第一次：dy/dx = 2x = 4
print(f"1.5  第1次 backward 后 x.grad={x.grad.item()}  (期望 4.0)")
y1 = x ** 2
y1.backward()                   # 第二次：又累加 4，变成 8（而不是重新算成 4）
print(f"1.5  第2次 backward 后 x.grad={x.grad.item()}  (变成 8.0 —— 累加了！)")
# 所以训练循环里每轮必须清零梯度，否则梯度越滚越大，参数更新方向就乱了


# ============================================================
# Part 2: 手写训练循环 —— 不用任何封装，看清每一步
# ============================================================
print("\n【Part 2】手写线性回归（不用 nn.Module）\n")

# --- 2.1 造数据：真实规律 y = 2x + 1，加噪声模拟真实世界 ---
x = torch.linspace(0, 1, 100).unsqueeze(1)
# linspace(0,1,100)：0 到 1 之间均匀取 100 个点，形状 (100,)
# unsqueeze(1)：在第 1 维插入维度 → (100, 1)
# 变二维是因为神经网络惯例：一行 = 一个样本，一列 = 一个特征
# 明天 DataLoader 的 batch 也是这个格式

y_true = 2 * x + 1 + 0.1 * torch.randn(x.shape)
# 2x+1 是真实规律（模型不知道，要自己学）
# 0.1*randn 是正态噪声，模拟真实数据的不完美

# --- 2.2 定义参数：这就是"模型"，线性回归只有 2 个未知数 ---
w = torch.randn(1, requires_grad=True)   # 斜率，随机初始化 —— 一开始是瞎猜的
b = torch.zeros(1, requires_grad=True)   # 截距，从 0 开始

print(f"训练前：w={w.item():.3f}  b={b.item():.3f}  (随机值，离 2 和 1 很远)")

# --- 2.3 前向传播 + 损失函数 ---
def forward(x):
    return w * x + b
# 用当前参数做预测。因为 w、b 带 requires_grad，这次运算会被记进计算图

def loss_fn(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()
# MSE 均方误差：误差 → 平方（负变正 + 重罚大错）→ 求平均 → 一个标量
# 训练的唯一目标：把这个数变小

# --- 2.4 训练循环（核心！五行逻辑，全世界的训练代码都是它）---
lr = 0.1                         # 学习率：每步迈多大
for epoch in range(200):         # 200 轮 = 把数据完整看 200 遍
    y_pred = forward(x)                  # ① 前向：算预测值
    loss = loss_fn(y_pred, y_true)       # ② 打分：预测得多差

    loss.backward()                      # ③ 反向：算出 ∂loss/∂w 和 ∂loss/∂b，存进 .grad
    # 为什么 -= 而不是 += ？梯度指向 loss「上升最快」的方向，
    # 我们要下山，所以朝反方向走（数学上：loss 变化量 ≈ grad × Δw = -lr·grad² < 0）

    with torch.no_grad():                # ④ 更新参数。no_grad：更新动作不需要被记进计算图
        w -= lr * w.grad                 # 新w = 旧w - 学习率 × 梯度
        b -= lr * b.grad

    w.grad.zero_()                       # ⑤ 清零梯度（对应 1.5 里的累加坑）
    b.grad.zero_()

    if epoch % 20 == 0:
        # .item()：把单元素张量转成普通 Python 数字，打印时不带 tensor() 壳
        print(f"  epoch={epoch:3d}  loss={loss.item():.4f}  w={w.item():.3f}  b={b.item():.3f}")

print(f"\n训练后：w={w.item():.3f} (期望≈2.0)  b={b.item():.3f} (期望≈1.0)  ✅")
# 模型从未被告知规律是 2x+1，仅凭数据自己学出了参数 —— 这就是机器学习


# ============================================================
# Part 3: nn.Module 版本 —— 和 Part 2 完全等价，只是封装了重复劳动
# ============================================================
print("\n【Part 3】nn.Module 版本（工业界写法）\n")

class LinearModel(nn.Module):           # 必须继承 nn.Module
    def __init__(self):
        super().__init__()              # 必须第一行：先初始化父类，参数才有地方登记
        self.linear = nn.Linear(1, 1)   # 1 维输入 → 1 维输出
        # nn.Linear 内部自动创建了 Part 2 里手写的 w 和 b，且自动 requires_grad=True

    def forward(self, x):               # 定义数据如何流过模型
        return self.linear(x)           # 内部计算：x·Wᵀ + b

model = LinearModel()

# --- 参数创建：替代手写 w、b ---
print("模型参数：")
for name, p in model.named_parameters():
    # 会打印 linear.weight (1,1) 和 linear.bias (1,)
    # 对照 Part 2：这就是你的 w 和 b，只是自动创建好了
    print(f"  {name}: shape={tuple(p.shape)}, requires_grad={p.requires_grad}")

# --- 损失函数：替代手写 ((pred-true)**2).mean() ---
loss_fn = nn.MSELoss()                  # 实例化一次，循环里反复用
# --- 优化器：替代手写 w -= lr*w.grad ---
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# model.parameters() 把所有可学习参数一次性交给优化器
# 模型有 1 亿个参数也是这一行 —— 不用再一个个点名
# SGD 就是 Part 2 的普通梯度下降；LLM 常用 AdamW，思路相同只是自适应调步长

for epoch in range(200):
    y_pred = model(x)                   # ① 前向。注意是 model(x) 不是 model.forward(x)
    loss = loss_fn(y_pred, y_true)      # ② 损失
    loss.backward()                     # ③ 反向（一模一样）

    optimizer.step()                    # ④ 更新所有参数（替代 w -= lr*w.grad）
    optimizer.zero_grad()               # ⑤ 清零所有梯度（替代 w.grad.zero_()）
    # 注意：用 optimizer 后不再需要 torch.no_grad()，step() 内部已处理

    if epoch % 20 == 0:
        print(f"  epoch={epoch:3d}  loss={loss.item():.4f}")

w_m = model.linear.weight.item()
b_m = model.linear.bias.item()
print(f"\n训练后：w={w_m:.3f}  b={b_m:.3f}")
print("和 Part 2 结果几乎一致 → 框架没有魔法，只是封装 ✅")


# ============================================================
# Part 4: 学习率实验 —— 亲手制造"收敛慢"和"梯度爆炸"
# ============================================================
print("\n【Part 4】学习率实验\n")

def train_with_lr(lr, epochs=200, verbose=False):
    """用指定学习率训练一次，返回最终 loss"""
    torch.manual_seed(42)
    m = LinearModel()
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for e in range(epochs):
        pred = m(x)
        l = criterion(pred, y_true)
        l.backward()
        opt.step()
        opt.zero_grad()
        if verbose and e % 40 == 0:
            print(f"    epoch={e:3d} loss={l.item():.4f}")
    return l.item()

print("a) lr=0.001（步子太小）→ 收敛极慢：")
final = train_with_lr(0.001, verbose=True)
print(f"   最终 loss={final:.4f}  → 200 轮走不完，参数还离最优很远\n")

print("b) lr=0.1（合适）→ 正常收敛：")
final = train_with_lr(0.1, verbose=True)
print(f"   最终 loss={final:.4f}  → 收敛良好\n")

print("c) lr=1.0（步子太大）→ 震荡/爆炸：")
final = train_with_lr(1.0, verbose=True)
print(f"   最终 loss={final:.4f}  → 注意 loss 变大甚至 inf/nan")
# 原因：梯度下降的下降保证依赖"一阶泰勒近似"，即步子必须足够小。
# 步子一大，近似失效，一步跨过谷底落到更高的坡上，loss 反而上升 → 正反馈发散


# ============================================================
# Part 5: 自测题（答案写在注释里，先自己答再看）
# ============================================================
print("\n【Part 5】自测\n")
print("Q1: nn.Linear(1,1) 里有几个可学习参数？")
print("A1: 2 个 —— 1 个 weight(1,1) + 1 个 bias(1,)")
print()
print("Q2: optimizer.zero_grad() 对应手写版的哪一行？")
print("A2: w.grad.zero_() / b.grad.zero_()，作用是防止梯度累加")
print()
print("Q3: 为什么必须有 super().__init__()？")
print("A3: 父类负责建立参数登记机制，不调它参数挂不上去，会直接报错")
print()
print("Q4: 为什么更新参数要 -= 而不是 +=？")
print("A4: 梯度指向 loss 上升最快方向；-= 即沿负梯度走，泰勒展开可证 loss 必然下降")
print("=" * 70)
