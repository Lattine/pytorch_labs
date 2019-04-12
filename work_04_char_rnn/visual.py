# -*- coding: UTF-8 -*-

import visdom
import numpy as np


class Visualizer:
    def __init__(self, env="default", **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        # 画的第几个数值，相当于横坐标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log = ""

    def re_init(self, env="default", **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)  # 再次初始化
        return self

    def plot(self, name, val):
        """ self.plot("loss", 1.01) """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([val]), X=np.array([x]), win=name, opts=dict(title=name), update=None if x == 0 else "append")
        self.index[name] = x + 1
