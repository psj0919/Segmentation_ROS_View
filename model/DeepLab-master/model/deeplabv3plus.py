import torch
import torch.nn as nn
import torch.nn.functional as F
from model.aspp_module import build_aspp
from model.decoder import build_decoder
from backbone.ResNet import build_backbone


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21, sync_bn=False, freeze_bn=False, pretrained=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8
        BatchNorm = nn.BatchNorm2d
        self.pretrained = pretrained
        self.backbone = build_backbone(backbone, output_stride, BatchNorm, self.pretrained)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feature = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feature)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


