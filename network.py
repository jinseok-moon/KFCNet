# Basic import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# todo: residual block not finished
class ResBlock(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, is_downsample):
        super(ResBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(),
            nn.Conv2d(mid_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

        def forward(self, x):
            out = self.residual(x)  # F(x)
            if is_downsample:
                out = out + x  # F(x) + x
            out = nn.ReLU(out)
            return out


# Firstly, modeling with ResNet50
class KFCNet(nn.Module):
    def __init__(self):
        super(KFCNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer2 = nn.Sequential(
            ResBlock(64, 64, 256, False),
            ResBlock(256, 64, 256, False),
            ResBlock(256, 64, 256, True)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)  # Flatten them for FC
        return x