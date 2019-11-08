# -*- coding: utf-8 -*-

# @Time    : 2019/11/8
# @Author  : Lattine

# ======================
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRCNN(nn.Module):
    def __init__(self, config, embedding_pretrained=None):
        super(TextRCNN, self).__init__()
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
                           batch_first=True)
        self.maxpool = nn.MaxPool1d(config.sequence_length)
        self.fc = nn.Linear(config.rnn_hidden_size * 2 + config.embedding_size, config.num_classes)

    def forward(self, x):
        embed = self.embedding(x)  # [B, seq, emb]
        H, _ = self.rnn(embed)  # [B, seq, hid*2]
        out = torch.cat((embed, H), dim=2)  # [B, seq, hid*2 + emb]
        out = F.relu(out)
        out = out.permute(0, 2, 1)  # [B, hid*2+emb, seq]
        out = self.maxpool(out).squeeze()  # [B, hid*2+emb]
        out = self.fc(out)
        return out
