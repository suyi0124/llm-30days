import torch
from torch import nn

from autogradlearning import y_true

# # t = torch.tensor([[1, 2],[3,4]])
# # print(t)
# #
# # r = torch.randn(3,4)
# # print(r)
# #
# # x3 = torch.zeros(3,4)
# # print(x3)
# #
# # x4 = torch.ones(3,4)
# # print(x4)
# #
# # x5 = torch.arange(start=0,end=10)
# # print(x5)
# #
# # x6 = x5.reshape(5,2)
# # print(x6)
# #
# # device = 'cuda' if torch.cuda.is_available() else 'no'
# # print(device)
# #
# # #矩阵乘法
# # a = torch.tensor([[1,2],[3,4]])
# # b = torch.tensor([[5,6],[7,8]])
# # r1 = torch.matmul(a, b)
# # print(r1)
#
# # house_size = 50
# # real_size = 100
# # #wx+b = y
# # w = 1.5
# # b = 10
# #
# # pre_price = w * house_size + b
# # error = pre_price - real_size
# #
# # dl_dw = 2 *(pre_price - real_size) *  house_size
# # dl_db = 2 * error
# #
# # print(dl_dw)
# # print(dl_db)
# def pytorch_autograd():
#     house_size = torch.tensor([50.0])
#     real_size  = torch.tensor([100.0])
#     w = torch.tensor([1.5],requires_grad=True)
#     b=  torch.tensor([10.0],requires_grad=True)
#
#     pre_size  = w*house_size+b
#     loss = torch.nn.MSELoss()(pre_size,real_size)
#
#     loss.backward()
#
#     return w.grad.item(),b.grad.item()
#
# if __name__ == '__main__':
#     print(pytorch_autograd())
#
#
# x = torch.linspace(0,1,100).unsqueeze(1)
#
# y_true = x * 2 + 1 + 0.1 * torch.randn(x.shape)
#
# w = torch.randn(1,requires_grad=True)
#
# b = torch.zeros(1,requires_grad=True)
#
# def forwward(x):
#     return w*x + b
#
# def loss_fn(y_true, y_pred):
#     return ((y_true - y_pred)**2).mean()
#
#
# lr = 0.1
# for epoch in range(200):
#     y_pred = forwward(x)
#     loss = loss_fn(y_true, y_pred)
#     loss.backward()
#     with torch.no_grad():
#         w -= lr * w.grad
#         b -= lr * b.grad
#     if epoch % 20 == 0:
#         print(f"epoch={epoch:3d} loss={loss.item():.4f} w={w.item():.3f} b={b.item():.3f}")

# x = torch.linspace(0, 1, 100).unsqueeze(1)
# y_true = 2 * x + 1+0.1*torch.randn(x.shape)
#
#
# class LinearModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(1, 1)
#
#     def forward(self, x):
#         return self.linear(x)
#
# model = LinearModel()
#
# loss_fn = nn.MSELoss()
#
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#
# for epoch in range(200):
#     y_pred = model(x)
#     loss = loss_fn(y_pred, y_true)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#
#     if epoch % 20 == 0:
#         print(f"epoch:{epoch}, loss:{loss.item():.4f}")
#
# w = model.linear.weight.item()
#
# b = model.linear.bias.item()
#
# print(f"w:{w:.3f}, b:{b:.3f}")

