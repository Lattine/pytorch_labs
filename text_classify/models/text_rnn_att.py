# -*- coding: utf-8 -*-

# @Time    : 2019/11/7
# @Author  : Lattine

# ======================
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRNNAtt(nn.Module):
    def __init__(self, config, embedding_pretrained=None):
        super(TextRNNAtt, self).__init__()
        self.config = config
        if embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)  # 0 for PAD

        self.rnn = nn.LSTM(input_size=config.embedding_size,
                           hidden_size=config.rnn_hidden_size,
                           num_layers=config.num_layers,
                           dropout=config.dropout,
                           bidirectional=True,
                           batch_first=True,
                           )
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.Tensor(config.rnn_hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc = nn.Linear(config.rnn_hidden_size * 2, config.att_hidden_size)
        self.out = nn.Linear(config.att_hidden_size, config.num_classes)

    def forward(self, x):
        out = self.embedding(x)
        H, _ = self.rnn(out)  # [batch_size, seq_len, hidden_size * num_direction]

        M = self.tanh1(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, 1)
        out = F.relu(out)
        out = self.fc(out)
        out = self.out(out)
        return out
