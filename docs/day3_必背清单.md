# Day 3 必背清单（就这 20 条，多一条都不背）

> 使用方法：早晚各朗读一遍，连背 3 天。每条背完立刻在代码里用一次（哪怕只是打出来看一眼）。
> 背诵只负责"知道"，动手负责"会"。两条腿缺一不可。

## A. 概念句（背原文）

1. 张量四属性：shape（形状）、dtype（类型）、device（在哪个设备）、requires_grad（要不要求导）
2. 图像张量四维：`(batch, 通道, 高, 宽)`，MNIST 是 `(64, 1, 28, 28)`
3. Dataset 管"一条数据怎么取"，DataLoader 管"怎么批量取、打乱、并行"
4. batch_size：一批几张图；epoch：全部数据过一遍；一批更新一次参数
5. MNIST：训练集 6 万张，测试集 1 万张，0-9 共 10 类，28×28 像素
6. Flatten 把 `(64,1,28,28)` 压成 `(64,784)`，因为 Linear 只吃二维 `(样本数, 特征数)`
7. Linear(784, 256) 参数量 = 784×256 + 256 ≈ 20 万（权重+偏置）
8. 没有激活函数，多层网络等价于一层（W2W1x = (W2W1)x），只能画直线边界
9. CrossEntropyLoss = softmax + 交叉熵，所以前面不能再加 softmax
10. logits = 未经 softmax 的原始分数，形状 (batch, 类别数)

## B. 训练五行循环（背到能默写）

11. 前向：`y_pred = model(x)`
12. 损失：`loss = loss_fn(y_pred, y)`
13. 反向：`loss.backward()`
14. 更新：`optimizer.step()`
15. 清零：`optimizer.zero_grad()`
    —— 口诀：**预测、打分、求导、更新、清零**

## C. 规则句（违反必出 bug）

16. backward 前必须清零梯度，否则梯度累加、更新方向错乱
17. 数据和模型必须在同一设备：模型 `.to(device)`，每个 batch 的 x、y 也要 `.to(device)`
18. 评估前 `model.eval()`（关训练行为）+ `with torch.no_grad():`（不建计算图），两者不可互替
19. 单元素张量变普通数用 `.item()`；单张图给 imshow 前先 `.squeeze()` 去掉大小为 1 的维度
20. 保存模型用 `torch.save(model.state_dict(), 路径)`，只存参数不存对象

---

## 不背清单（明确不背，用到就查文档）

- matplotlib 各参数（figsize、dpi、cmap 的可选值）
- DataLoader 的冷门参数（num_workers、pin_memory）
- torch.optim 各优化器的内部区别
- torchvision.transforms 的其他变换

## 背完的自检方式（每周做一次，5 分钟）

1. 白纸默写五行循环 → 对照代码
2. 口头回答：Linear(784,256) 几个参数？为什么删 ReLU 会掉点？
3. 闭卷写出：加载 MNIST → 建 DataLoader → 定义模型 的骨架（方法名对即可，参数错了不算输）
