# Basic import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Firstly, modeling with ResNet50
class KFCNet(nn.Module):
    def __init__(self):
        super(KFCNet, self).__init__()
        # b, 3, 255, 255
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, padding=4),  # b, 96, 55, 55
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=(4, 4))
        )

        self.layer2 = nn.Sequential(
            nn.Linear(in_features=32 * 6 * 6, out_features=128),
            nn.Linear(in_features=128, out_features=27)  # number of food classes
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)  # Flatten them for FC
        x = self.layer2(x)
        return x