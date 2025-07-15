import torch
from torch import nn
from model.ECA_module import ECA



class DA_ECA(nn.Module):
    def __init__(self, k):
        super(DA_ECA, self).__init__()
        self.k = k
        self.eca = ECA(self.k)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, channel, width, height = x.shape
        #
        proj_query = self.eca(x).view(batch_size, channel, -1)
        proj_key = self.eca(x).view(batch_size, channel, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = self.eca(x).view(batch_size, channel, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out

if __name__=='__main__':
    x = torch.randn(1, 3, 256, 256)
    DA_ECA = DA_ECA()
    out = DA_ECA(x)
