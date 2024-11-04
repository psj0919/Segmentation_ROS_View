import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Segmenter.utils import padding, unpadding
from timm.models.layers import trunc_normal_

class Segmenter(nn.Module):
    def __init__(self, encoder, decoder, n_cls):
        super().__init__()
        self.n_cls = n_cls
        self.path_size = encoder.path_size
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.path_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_features = True)

        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode= "bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks
    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)


    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
