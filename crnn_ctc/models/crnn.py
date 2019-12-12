# -*- coding: utf-8 -*-

# @Time    : 2019/12/9
# @Author  : Lattine

# ======================
import os
import torch.nn as nn
import torch.nn.functional as F


class Config:
    # BASE_URL = os.path.abspath(os.path.dirname(os.getcwd()))
    BASE_URL = r"E:\tmp\ch_crnn"  # for local test

    # --------- 构造数据集 -----------
    flush_examples = 1000
    height = 32  # 图片的高度, 宽度会根据度自适应
    train_image_label_file = os.path.join(BASE_URL, "data", "train_attachments.txt")
    train_image_prefix = os.path.join(BASE_URL, "data", "images")
    train_lmdb = os.path.join(BASE_URL, "data", "train_lmdb")
    eval_image_label_file = os.path.join(BASE_URL, "data", "eval_attachments.txt")
    eval_image_prefix = os.path.join(BASE_URL, "data", "images")
    eval_lmdb = os.path.join(BASE_URL, "data", "eval_lmdb")
    chars_file = os.path.join(BASE_URL, "data", "chars.txt")

    # ----------- Train -----------
    model_name = "crnn"
    save_path = os.path.join(BASE_URL, "ckpt")
    saved_model = os.path.join(save_path, model_name + ".pth")  # 模型保存的路径及名称
    epoches = 10000
    batch_size = 16
    workers = 0
    lr = 1e-3
    min_loss = float("inf")
    early_stop_thresh = 50  # 提前终止的轮数

    def __init__(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        with open(self.chars_file, encoding="utf-8") as fr:
            self.alphabets = fr.read().strip()
            # self.alphabets = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/.%"


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.ks = [3, 3, 3, 3, 3, 3, 2]
        self.ps = [1, 1, 1, 1, 1, 1, 0]
        self.ss = [1, 1, 1, 1, 1, 1, 1]
        self.nm = [64, 128, 256, 256, 512, 512, 512]

        self.cnn = nn.Sequential()
        self.add_cnn_layer(0)
        self.cnn.add_module(f"pooling{0}", nn.MaxPool2d(2, 2))  # [64, H/2, W/2]
        self.add_cnn_layer(1)
        self.cnn.add_module(f"pooling{1}", nn.MaxPool2d(2, 2))  # [128, H/4, W/4]
        self.add_cnn_layer(2, bn=True)
        self.add_cnn_layer(3)
        self.cnn.add_module(f"pooling{2}", nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # [256, H/8, W/4]
        self.add_cnn_layer(4, bn=True)
        self.add_cnn_layer(5)
        self.cnn.add_module(f"pooling{3}", nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # [512, H/16, W/4]
        self.add_cnn_layer(6, bn=True)

    def add_cnn_layer(self, n_layer, bn=False, nc=1):
        ch_in = nc if n_layer == 0 else self.nm[n_layer - 1]
        ch_out = self.nm[n_layer]
        self.cnn.add_module(f"conv{n_layer}", nn.Conv2d(ch_in, ch_out, kernel_size=self.ks[n_layer], stride=self.ss[n_layer], padding=self.ps[n_layer]))
        if bn:
            self.cnn.add_module(f"batchnorm{n_layer}", nn.BatchNorm2d(ch_out))
        self.cnn.add_module(f"relu{n_layer}", nn.ReLU(True))

    def forward(self, x):
        out = self.cnn(x)
        return out


class BiLSTM(nn.Module):
    def __init__(self, n_in, hidden_unit, n_out):
        super(BiLSTM, self).__init__()
        self.rnn = nn.LSTM(n_in, hidden_unit, bidirectional=True)
        self.fc = nn.Linear(hidden_unit * 2, n_out)

    def forward(self, x):
        x, _ = self.rnn(x)
        T, B, h = x.size()
        t_rec = x.view(T * B, h)
        out = self.fc(t_rec)
        out = out.view(T, B, -1)
        return out


class Model(nn.Module):
    def __init__(self, class_num, hidden_unit=256):
        super(Model, self).__init__()
        self.cnn = nn.Sequential()
        self.cnn.add_module("VGG16", VGG16())
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
        out = F.log_softmax(x, dim=2)
        return out
