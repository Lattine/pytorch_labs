# -*- coding: utf-8 -*-

# @Time    : 2019/10/11
# @Author  : Lattine

# ======================
import collections

import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt


class StrLabelConverter:
    """ 文本编码 """

    def __init__(self, alphabets, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:  # 忽略大小写
            alphabets = alphabets.lower()
        self.alphabets = alphabets + "-"  # alphabets[-1], 为了CTC分割，插入的BLANK标志
        self.dict = {}
        for i, c in enumerate(alphabets):
            self.dict[c] = i + 1  # 保留 0 给BLANK标志

    def encode(self, text):
        if isinstance(text, str):
            text = [self.dict[c.lower() if self._ignore_case else c] for c in text]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = "".join(text)
            text, _ = self.encode(text)
        return torch.IntTensor(text), torch.IntTensor(length)

    def decode(self, t, length, raw=False):
        if length.numel() == 1:  # 只有一行文本
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return "".join([self.alphabets[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabets[t[i] - 1])
            return "".join(char_list)
        else:
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                lth = length[i]
                texts.append(self.decode(t[index:index + lth], torch.IntTensor([lth]), raw=raw))
                index += lth
            return texts


class averager(object):
    """计算 `torch.Variable` ， `torch.Tensor`的平均值. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def draw_loss_plot(train_loss_list=[], test_loss_list=[], loss_image_path=None):
    x1 = range(0, len(train_loss_list))
    x2 = range(0, len(test_loss_list))
    y1 = train_loss_list
    y2 = test_loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('train loss vs. iterators')
    plt.ylabel('train loss')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('test loss vs. iterators')
    plt.ylabel('test loss')
    if loss_image_path:
        plt.savefig(loss_image_path)
