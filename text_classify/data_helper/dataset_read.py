# -*- coding: utf-8 -*-

# @Time    : 2019/10/24
# @Author  : Lattine

# ======================
"""
用于创建Dataset的辅助函数，不同的数据集只要实现该文件读入的函数，即可复用Dataset。
"""

import pandas as pd


# ------------------ O2O商铺食品安全相关评论发现 ------------------
def read_file(path):
    raw = pd.read_csv(path)
    raw["label"] = raw["label\tcomment"].apply(lambda x: x.strip().split("\t")[0])
    raw["comment"] = raw["label\tcomment"].apply(lambda x: x.strip().split("\t")[1])
    raw["cutted_comment"] = raw["comment"].apply(lambda x: list(x))
    x = raw["cutted_comment"].values.tolist()
    y = raw["label"].values.tolist()
    return x, y


if __name__ == '__main__':
    from config import Config

    cfg = Config()
    read_file(cfg.train_data)
