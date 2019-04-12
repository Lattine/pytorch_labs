# -*- coding: UTF-8 -*-

import pickle
import numpy as np
from collections import Counter
import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, path, seq_len, text2arr):
        with open(path, encoding="utf-8") as f:
            text = f.read().replace("\n", "")  # 读取文件，并删除换行符
            # text = text.replace("，", "").replace("。", "")  # 删除标点符号，可选
        num_seq = len(text) // seq_len  # 以seq_len划分文本
        text = text[:num_seq * seq_len]  # 删除多余部分
        data = text2arr(text)  # 转换为Index
        data = data.reshape((num_seq, -1))  # 将Index向量转换为二维张量，其中行为num_seq, 列为seq_len
        self.num_seq = num_seq  # data有多少行
        self.data = torch.from_numpy(data)  # 转换为tensor

    def __getitem__(self, ix):
        item = self.data[ix, :]  # 取出第ix条数据
        return item

    def __len__(self):
        return self.num_seq  # 数据集长度


class TextConverter:
    def __init__(self, path, vocab_size=5000):
        with open(path, encoding="utf-8") as f:
            text = f.readlines()  # 读取所有行
        word_list = [w for s in text for w in s]  # 转为List, ['今','天','天'....]
        word_counter = Counter(word_list)  # 统计每个字符出现的频率
        vocab = []
        for word, counter in word_counter.most_common(vocab_size):  # 只统计前面高频的词
            vocab.append(word)  # 放入词袋
        word2ix = {word: i for i, word in enumerate(vocab)}  # 字到向量的字典表
        ix2word = {i: word for i, word in enumerate(vocab)}  # 向量到字的字典表
        if ".txt" in path:
            with open(path[:-4] + ".pkl", "wb") as fp:  # 序列化，可选
                pickle.dump(vocab, fp)
                pickle.dump(word2ix, fp)
                pickle.dump(ix2word, fp)
        self.vocab = vocab
        self.word2ix = word2ix
        self.ix2word = ix2word

    @property
    def vocab_size(self):
        """ 词袋长度，+1是包括未登录词<unk> """
        return len(self.vocab) + 1

    def word2int(self, w):
        """ 字到索引 """
        if w in self.word2ix:
            return self.word2ix[w]
        else:
            return len(self.vocab)

    def int2word(self, ix):
        """ 索引到字 """
        if ix == len(self.vocab):
            return "<unk>"
        if 0 <= ix < len(self.vocab):
            return self.ix2word[ix]
        else:
            raise ValueError("out of index")

    def text2arr(self, text):
        """ 字向量到索引向量 """
        arr = []
        for w in text:
            arr.append(self.word2int(w))
        return np.array(arr)

    def arr2text(self, arr):
        """ 索引向量到字向量 """
        word = []
        for ix in arr:
            word.append(self.int2word(ix))
        return word
