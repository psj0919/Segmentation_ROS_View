import torch
from torch import nn
from math import log


class ECA(nn.Module):
    def __init__(self, k):
        super(ECA, self).__init__()
        self.k = k
        self.gamma = 2
        self.b = 1
        self.conv1d = nn.Conv1d(1, 1, kernel_size=self.k, padding=int(self.k/2), bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv1d(out.squeeze(-1).transpose(-1, -2))
        out = out.transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)

        return x * out.expand_as(x)


if __name__== '__main__':
    x = torch.randn(1, 3, 256, 256)
    ECA = ECA()
    out = ECA(x)
