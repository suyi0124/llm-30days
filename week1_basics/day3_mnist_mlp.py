import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

transform = transforms.ToTensor()

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=True)

example = iter(train_loader)
images, labels = next(example)

print(f"一个 batch 的图像形状: {images.shape}")   # 64张图, 1个颜色通道, 28x28像素
print(f"对应的标签: {labels[:9].tolist()}")

fig,axes = plt.subplots(3,3,figsize=(6,6))

for i,ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(),cmap="gray")
    ax.set_title(f"标签：{labels[i].item()}")
    ax.axis('off')


plt.suptitle("MNIST 样本预览")
plt.savefig("day3_samples.png",dpi=120)
print("样本图已保存: day3_samples.png")

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,256),
            nn.ReLU(),
            nn.Linear(256,10),
        )

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
print(model)
print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ============ 第 4 步 · 训练（epoch 套 batch 双层循环） ============

for epoch in range(5):
    model.train()
    total_loss = 0.0
    correct = 0

    for x,y in train_loader:
        x.to(device)
        y.to(device)

        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        correct +=  (y_pred.argmax(1) == y).float().sum().item()

    train_acc = correct / len(train_ds)

    model.eval()
    test_correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_correct += (pred.argmax(1) == y).float().sum().item()
    test_acc = test_correct / len(test_ds)

    print(f"epoch {epoch + 1}/5 | 训练loss={total_loss / len(train_loader):.4f} "
          f"| 训练acc={train_acc:.4f} | 测试acc={test_acc:.4f}")


# ============ 第 5 步 · 保存模型 ============

torch.save(model.state_dict(), "day3_mlp.pth")
print("模型已保存: day3_mlp.pth")

