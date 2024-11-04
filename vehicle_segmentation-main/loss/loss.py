from distutils.version import LooseVersion
import torch.nn.functional as F
import torch
import torch.nn as nn


class cross_entropy2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(cross_entropy2d, self).__init__()
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        # input: (n, c, h, w), target: (n, h, w)
        n, c, h, w = input.size()
        if LooseVersion(torch.__version__) < LooseVersion('0.3'):
            log_p = F.log_softmax(input)
        else:
            log_p = F.log_softmax(input, dim=1)

        # log_p = (n*h*w, c)
        log_p = log_p.transpose(dim0=1, dim1=2).transpose(2, 3).contiguous()
        log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, weight=self.weight, reduction='sum')
        if self.size_average:
            loss /= mask.data.sum()

        return loss
