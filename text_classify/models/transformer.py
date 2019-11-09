# -*- coding: utf-8 -*-

# @Time    : 2019/11/9
# @Author  : Lattine

# ======================
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, config, embedding_pretrained=None):
        super(Transformer, self).__init__()
        if embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)  # 0 for PAD

        self.position_embedding = PositionalEncoding(config.embedding_size, config.sequence_length, config.dropout, config.device)
        encoder = Encoder(config.model_dim, config.num_heads, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([copy.deepcopy(encoder) for _ in range(config.num_encoders)])
        self.fc1 = nn.Linear(config.sequence_length * config.model_dim, config.num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out = self.position_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class Encoder(nn.Module):
    def __init__(self, model_dim, num_heads, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionWrisFeedForward(model_dim, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, embed, sequence_length, dropout, device):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(sequence_length)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        assert model_dim % num_heads == 0
        self.head_dim = model_dim // self.num_heads
        self.fc_Q = nn.Linear(model_dim, num_heads * self.head_dim)
        self.fc_K = nn.Linear(model_dim, num_heads * self.head_dim)
        self.fc_V = nn.Linear(model_dim, num_heads * self.head_dim)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(num_heads * self.head_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        B = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(B * self.num_heads, -1, self.head_dim)
        K = K.view(B * self.num_heads, -1, self.head_dim)
        V = V.view(B * self.num_heads, -1, self.head_dim)
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale)
        context = context.view(B, -1, self.head_dim * self.num_heads)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


class PositionWrisFeedForward(nn.Module):
    def __init__(self, model_dim, hidden, dropout=0.0):
        super(PositionWrisFeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden)
        self.fc2 = nn.Linear(hidden, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out
