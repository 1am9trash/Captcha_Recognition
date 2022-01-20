import torch
import torch.nn as nn
import numpy as np

class CAPTCHA_MODEL(nn.Module):
    def __init__(self):
        super(CAPTCHA_MODEL, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )
        self.flatten = nn.Flatten()
        self.output = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 144)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.output(x)
        return x
