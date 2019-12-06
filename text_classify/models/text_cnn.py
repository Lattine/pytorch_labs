# -*- coding: utf-8 -*-

# @Time    : 2019/10/24
# @Author  : Lattine

# ======================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    BASE_URL = os.path.abspath(os.path.dirname(os.getcwd()))
    rebuild = False  # 是否重新构建模型

    # ---------- 数据集 ----------
    train_data = os.path.join(BASE_URL, "data", "cmt", "train.csv")  # 训练数据
    eval_data = os.path.join(BASE_URL, "data", "cmt", "train.csv")  # 验证数据
    test_data = os.path.join(BASE_URL, "data", "cmt", "train.csv")  # 预测数据
    output_path = os.path.join(BASE_URL, "data", "cmt", "output")  # 字典等数据集输出目录
    stopwords = os.path.join(BASE_URL, "data", "stopwords.txt")  # 停用词文件
    word_vectors = os.path.join(BASE_URL, "data", "sgns.weibo.char")  # 词向量文件

    # ---------- 模型 -----------
    model_name = "text_cnn"
    save_path = os.path.join(BASE_URL, "ckpt")
    saved_model = os.path.join(save_path, model_name + ".pth")  # 模型保存的路径及名称
    num_classes = 5  # 目标的类别数目
    sequence_length = 100  # 需要调用data_helper中的data_analysis.py，对不同的数据集，具体测试
    vocab_size = 10000000  # 词袋大小，具体会根据数据集自动调整，这里设置一个超大值即可
    embedding_size = 300  # 词向量的维度，如果有预加载词向量，应与预加载的词向量维度相等
    filter_sizes = (2, 3, 4)  # 卷积核尺寸
    num_filters = 100  # 卷积核个数
    hidden_size = 128
    dropout = 0.5

    # ---------- Train ----------
    lr = 1e-3
    batch_size = 64
    epoches = 1000
    min_loss = float("inf")  # 预先设置的阈值
    early_stop_thresh = 50  # 提前终止的轮数

    # ---------- Predict ----------
    predict_result = os.path.join(BASE_URL, "data", "result.csv")  # 预测输出文件的路径


class Model(nn.Module):
    def __init__(self, config, embedding_pretrained=None):
        super(Model, self).__init__()
        self.config = config

        if embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=True)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)  # 0 for PAD
        self.convs = nn.ModuleList(  # 并列的CONVs
            # Input: (B, C, H, W)
            # Conv2d: (in_channels, out_channels, kernel_size)
            [nn.Conv2d(1, config.num_filters, (k, config.embedding_size)) for k in config.filter_sizes]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def _conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # [B, in, H1, W1] => [B, out, H2, 1] => [B, out, H2]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # [B, out, H] => [B, out], 其中kernel_size = x.size(2)
        return x

    def forward(self, x):
        out = self.embedding(x)  # [B, seq, emb]
        out = out.unsqueeze(1)  # [B, 1, seq, emb]
        out = torch.cat([self._conv_and_pool(out, conv) for conv in self.convs], 1)  # [B, out*3]
        out = self.dropout(out)
        out = self.fc(out)
        return out
