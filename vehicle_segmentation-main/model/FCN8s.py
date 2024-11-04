import torch
import torch.nn as nn
import numpy as np

class FCN8s(nn.Module):
    def __init__(self, num_class = 20):
        super().__init__()
        self.model_name = 'FCN8s'
        self.n_class = num_class

        # conv1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels= 64, kernel_size=3, padding=100),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels= 64, kernel_size= 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # conv2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels= 64, out_channels= 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels= 128, kernel_size= 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # conv3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels= 128, out_channels= 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels= 256, kernel_size= 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # conv4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # conv5
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(in_channels=4096, out_channels=self.n_class, kernel_size=1),
        )

        # Upsampling & 1by1 Conv Code
        self.conv3_1 = nn.Conv2d(256, self.n_class, 1)
        self.conv4_1 = nn.Conv2d(512, self.n_class,1)

        self.upscale = nn.ConvTranspose2d(self.n_class, self.n_class, 16, 8)
        self.upscale4 = nn.ConvTranspose2d(self.n_class, self.n_class, 4, 2)
        self.upscale5 = nn.ConvTranspose2d(self.n_class, self.n_class, 4, 2)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[1]
                factor = (n+1) // 2
                if n % 2 == 1:
                    center = factor - 1
                else:
                    center = factor - 0.5
                og = np.ogrid[:n, :n]
                weights_np = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
                m.weight.data.copy_(torch.from_numpy(weights_np))

    def forward(self, x):
        # Conv Block
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        conv6 = self.classifier(conv5)

        # Conv6 Upsampling
        upscale6 = self.upscale5(conv6)
        # Conv4 1X1 Conv
        scale4 = self.conv4_1(conv4)
        scale4 = scale4[:, :, 5:5+upscale6.size()[2], 5:5+upscale6.size()[3]].contiguous()
        # Conv6 + Conv4
        scale4 += upscale6

        # Conv3 1X1 Conv
        scale3 = self.conv3_1(conv3)
        # Upscale Conv6 + Conv4
        upscale4 = self.upscale4(scale4)
        scale3 = scale3[:, :, 9:9+upscale4.size()[2], 9:9+upscale4.size()[3]].contiguous()
        scale3 += upscale4

        out = self.upscale(scale3)
        out = out[:, :, 31:31+x.size()[2], 31:31+x.size()[3]].contiguous()
        return out



if __name__=='__main__':
    x = torch.randn(1, 3, 244, 244)
    model = FCN8s(num_class = 20)
    out = model(x)

