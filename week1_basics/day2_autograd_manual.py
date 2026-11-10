import torch
from torch.autograd import backward

x = torch.linspace(0, 1, 100).unsqueeze(1)
y_true = 2 * x + 1 + 0.1 * torch.randn(x.shape)
