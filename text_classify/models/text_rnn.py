# -*- coding: utf-8 -*-

# @Time    : 2019/11/6
# @Author  : Lattine

# ======================
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRNN(nn.Module):
    def __init__(self, config, embedding_pretrained=None):
        super(TextRNN, self).__init__()
        self.config = config

        if embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)  # 0 for PAD

        self.rnn = nn.LSTM(input_size=config.embedding_size,
                           hidden_size=config.hidden_size,
                           num_layers=config.num_layers,
                           bidirectional=True,
                           batch_first=True,
                           dropout=config.dropout
                           )
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        out = self.embedding(x)  # [B, seq, emb]
        out, _ = self.rnn(out)
        out = self.fc(out[:, -1, :])  # # 最后时刻的 hidden state
        return out
