import numpy as np
import torch.nn as nn
import cv2
import torch

class gamma_correction_sj(nn.Module):
    def __init__(self):
        super(gamma_correction_sj, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.conv1(img)
        out = self.gap(out)
        out = out.view(1, -1)
        gamma = self.fc(out).squeeze()
        gamma = gamma.item()

        corrected = img ** gamma
        return corrected, gamma
