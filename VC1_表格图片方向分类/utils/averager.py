# -*- coding: utf-8 -*-

# @Time    : 2019/10/16
# @Author  : Lattine

# ======================
import torch


class AveragerMeter:
    """计算 `torch.Variable` ， `torch.Tensor`的平均值. """

    def __init__(self):
        self.reset()

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / self.n_count
        return res

    def add(self, v):
        if isinstance(v, torch.autograd.Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()
        self.n_count += count
        self.sum += v
