import torch
import torch.nn as nn
import torch.nn.functional as F
from model.aspp_module import build_aspp
from model.decoder import build_decoder
from model.DA_ECA_module import DA_ECA
from math import log


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21, sync_bn=False, freeze_bn=False, pretrained=False, deploy=False, attention='backbone'):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8
        BatchNorm = nn.BatchNorm2d
        self.attention = attention
        self.deploy = True
        self.pretrained = pretrained
        if self.attention =='backbone':
            from backbone.RepVGG_ResNet import build_backbone
            print("load_resnet_ECA_backbone")
            self.backbone = build_backbone(backbone, BatchNorm, self.deploy)
        elif self.attention =='bottleneck1':
            from backbone.RepVGG_Resnet_bottleneck1 import build_backbone
            self.backbone = build_backbone(backbone, BatchNorm, self.deploy)        
            print("load_resnet_ECA_bottleneck1")
        elif self.attention =='bottleneck2':
            from backbone.RepVGG_Resnet_bottleneck2 import build_backbone
            self.backbone = build_backbone(backbone, BatchNorm, self.deploy) 
            print("load_resnet_ECA_bottleneck2")
        elif self.attention =='bottleneck3':
            from backbone.RepVGG_Resnet_bottleneck3 import build_backbone
            self.backbone = build_backbone(backbone, BatchNorm, self.deploy) 
            print("load_resnet_ECA_bottleneck3")
        elif self.attention =='bottleneck4':
            from backbone.RepVGG_Resnet_bottleneck4 import build_backbone
            self.backbone = build_backbone(backbone, BatchNorm, self.deploy)                                 
            print("load_resnet_ECA_bottleneck4")

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


