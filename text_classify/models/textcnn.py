# -*- coding: utf-8 -*-

# @Time    : 2019/10/24
# @Author  : Lattine

# ======================
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, config, embedding_pretrained=None):
        super(TextCNN, self).__init__()
        self.config = config

        if embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
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
