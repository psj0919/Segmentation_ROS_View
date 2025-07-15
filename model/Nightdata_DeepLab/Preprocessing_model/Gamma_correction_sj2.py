import numpy as np
import torch.nn as nn
import cv2
import torch

class gamma_correction(nn.Module):
    def __init__(self):
        super(gamma_correction, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, bias=False),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
               
    def forward(self, img):
        out = self.conv(img)
        out = self.fc(out)
        gamma = out.view(-1, 1, 1, 1)

        corrected = img ** gamma
        return corrected, gamma
