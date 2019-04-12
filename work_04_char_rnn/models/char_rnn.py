# -*- coding: UTF-8 -*-

import torch.nn as nn
from torch.autograd import Variable


class CharRNN(nn.Module):
    name = "char_rnn"  # 模型名称，用于保存模型之用

    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.5):
        super(CharRNN, self).__init__()  # 继承，调用父初始化函数
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # 词向量计算
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2, dropout=dropout)  # LSTM计算
        self.linear = nn.Linear(self.hidden_dim, vocab_size)  # 全连接层计算

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()  # 动态计算序列长度，批次
        if hidden is None:  # 首次，需要初始化隐藏单元
            h0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            h0, c0 = Variable(h0), Variable(c0)  # 包装为Variable
        else:  # 再次，复用隐藏单元
            h0, c0 = hidden
        embeds = self.embeddings(input)  # (seq_len, batch_size, embedding_dim)
        output, hidden = self.lstm(embeds, (h0, c0))  # (seq_len, batch_size, embedding_dim)
        out = self.linear(output.view(seq_len * batch_size, -1))  # (seq_len*batch_size, vocab_size)
        return out, hidden
