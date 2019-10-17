# -*- coding: utf-8 -*-

# @Time    : 2019/10/16
# @Author  : Lattine

# ======================
import torch.nn as nn
import torch.nn.functional as F


class TableClassifyNet(nn.Module):
    def __init__(self):
        super(TableClassifyNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pooling1 = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.pooling2 = nn.MaxPool2d(4, 4)
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        self.pooling3 = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(16 * 16 * 8, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.pooling1(F.relu(self.conv1(x)))
        x = self.pooling2(F.relu(self.conv2(x)))
        x = self.pooling3(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 16 * 8)
        x = F.dropout(F.relu(self.fc1(x)), 0.5)
        x = self.fc2(x)
        return x
