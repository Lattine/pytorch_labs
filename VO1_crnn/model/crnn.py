# -*- coding: utf-8 -*-

# @Time    : 2019/10/10
# @Author  : Lattine

# ======================
import torch.nn as nn
import torch.nn.functional as F


class VGG_16(nn.Module):
    def __init__(self):
        super(VGG_16, self).__init__()

        self.ks = [3, 3, 3, 3, 3, 3, 2]
        self.ps = [1, 1, 1, 1, 1, 1, 0]
        self.ss = [1, 1, 1, 1, 1, 1, 1]
        self.nm = [64, 128, 256, 256, 512, 512, 512]

        self.cnn = nn.Sequential()
        self.add_cnn_layer(0)
        self.cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        self.add_cnn_layer(1)
        self.cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        self.add_cnn_layer(2, bn=True)
        self.add_cnn_layer(3)
        self.cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        self.add_cnn_layer(4, bn=True)
        self.add_cnn_layer(5)
        self.cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        self.add_cnn_layer(6, True)  # 512x1x16

    def add_cnn_layer(self, n_layer, bn=True, nc=1):
        n_in = nc if n_layer == 0 else self.nm[n_layer - 1]
        n_out = self.nm[n_layer]
        self.cnn.add_module(f"conv{n_layer}", nn.Conv2d(n_in, n_out, self.ks[n_layer], self.ss[n_layer], self.ps[n_layer]))
        if bn:
            self.cnn.add_module(f"batchnorm{n_layer}", nn.BatchNorm2d(n_out))
        self.cnn.add_module(f"relu{n_layer}", nn.ReLU(True))

    def forward(self, x):
        x = self.cnn(x)
        return x


class BiLSTM(nn.Module):
    def __init__(self, n_in, hidden_unit, n_out):
        super(BiLSTM, self).__init__()
        self.rnn = nn.LSTM(n_in, hidden_unit, bidirectional=True)
        self.embedding = nn.Linear(hidden_unit * 2, n_out)

    def forward(self, x):
        x, _ = self.rnn(x)
        T, B, h = x.size()
        t_rec = x.view(T * B, h)
        out = self.embedding(t_rec)
        out = out.view(T, B, -1)
        return out


class CRNN(nn.Module):
    def __init__(self, class_num, hidden_unit=256):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential()
        self.cnn.add_module("VGG_16", VGG_16())
        self.rnn = nn.Sequential(
            BiLSTM(512, hidden_unit, hidden_unit),
            BiLSTM(hidden_unit, hidden_unit, class_num)
        )

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        assert h == 1  # the height of conv must be 1
        x = x.squeeze(2)  # remove h dimension => B * 512 * width
        x = x.permute(2, 0, 1)  # [w,b,c] = [seq_len, batch, input_size]
        x = self.rnn(x)
        return x
