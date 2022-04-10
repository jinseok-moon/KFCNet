# Basic import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class BasicBlock(nn.Module):
    mul = 1

    def __init__(self, in_dim, out_dim, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)

        # stride = 1, padding = 1이므로, 너비와 높이는 항시 유지됨
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

        # x를 그대로 더해주기 위함
        self.shortcut = nn.Sequential()

        # 만약 size가 안맞아 합연산이 불가하다면, 연산 가능하도록 모양을 맞춰줌
        if stride != 1:  # x와
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_dim)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # F(x) + x
        out = F.relu(out)
        return out


class BottleNeck(nn.Module):
    # 논문의 구조를 참고하여 mul 값은 4로 지정, 즉, 64 -> 256
    mul = 4

    def __init__(self, in_dim, out_dim, stride=1):
        super(BottleNeck, self).__init__()

        # 첫 Convolution은 너비와 높이 downsampling
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)

        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.conv3 = nn.Conv2d(out_dim, out_dim * self.mul, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_dim * self.mul)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_dim != out_dim * self.mul:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim * self.mul, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_dim * self.mul)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
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

        self.layer3 = nn.Sequential(
            ResBlock(256, 128, 512, False),
            ResBlock(512, 128, 512, False),
            ResBlock(256, 64, 256, True)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out