# -*- coding: utf-8 -*-

# @Time    : 2019/10/24
# @Author  : Lattine

# ======================
import re
import pandas as pd
import jieba


def read_file(path, N=3):
    """ 外汇利率短新闻 """
    raw = pd.read_csv(path)
    raw["content"] = raw["content"].apply(lambda x: re.sub(r"【.*?】|（.*?）|\d+|%|\.", "", x))
    raw["content"] = raw["content"].apply(lambda s: [s[i:i + n] for n in range(1, N + 1) for i in range(len(s) - n + 1)])
    x = raw["content"].values.tolist()
    raw["label"] = raw["label"].apply(lambda x: str(x))
    y = raw["label"].values.tolist()
    return x, y


if __name__ == '__main__':
    from config import Config

    cfg = Config()
    read_file(cfg.train_data)
