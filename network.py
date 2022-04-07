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
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)  # Flatten them for FC
        return x