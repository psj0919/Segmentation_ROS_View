import torch
import torch.nn as nn
from model.swin_transformer import *
from model.decoder.uper_head import *

class SwinTransformer_UperNet(nn.Module):
    def __init__(self,img_size=224, embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24), num_classes=21):
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone =  SwinTransformer(
            img_size=224, patch_size=4, embed_dim=self.embed_dim, depths=depths, num_heads=num_heads, num_classes=num_classes)
        self.pretrained = False

        if self.pretrained:
            path ='/storage/sjpark/vehicle_data/Pretrained_SwinTransformer/swin_large_patch4_window7_224_22k.pth'
            ckpt = torch.load(path, map_location='cpu')
            state = ckpt['model']  # OrderedDict
            for k in ['head.weight', 'head.bias']:
                if k in state:
                    state.pop(k)
            try:
                self.backbone.load_state_dict(state, strict=False)
                print("Success load weight")
            except:
                print("Error load weight")

        self.decoder = UPerHead(in_channels=[self.embed_dim*2**0, self.embed_dim*2**1, self.embed_dim*2**2, self.embed_dim*2**3], channels=512, num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone.forward_backbone(x)
        output = self.decoder(feats)

        return output

