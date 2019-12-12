# -*- coding: utf-8 -*-

# @Time    : 2019/12/10
# @Author  : Lattine

# ======================
import collections

import torch


class StrLabelConverter:
    """文本编码"""

    def __init__(self, alphabets, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
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
