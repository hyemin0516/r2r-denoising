from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, base, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(base, base * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(base * 2, base * 2, 3, padding=1)

        self.deconv1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv4 = nn.Conv2d(base, base, 3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(base, in_ch, 2, stride=2)
        self.conv5 = nn.Conv2d(in_ch, in_ch, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.deconv1(x)
        x = F.relu(self.conv4(x))
        x = self.deconv2(x)
        x = self.conv5(x)
        return x
