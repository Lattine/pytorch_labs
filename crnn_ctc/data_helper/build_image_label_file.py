# -*- coding: utf-8 -*-

# @Time    : 2019/12/9
# @Author  : Lattine

# ======================
import os
from models import Config

cfg = Config()
raw_uri = r"E:\BaiduNetdiskDownload\crnn_attention"

ch_test = os.path.join(raw_uri, "ch_test.txt")
with open(ch_test, encoding="utf-8") as fr, open(cfg.eval_image_label_file, "w", encoding="utf-8") as fw:
    for line in fr:
        segs = line.strip().split(" ")
        _, fn = os.path.split(segs[0])
        fw.write(f"{fn}\t{segs[1]}\n")

ch_train = os.path.join(raw_uri, "ch_train.txt")
with open(ch_train, encoding="utf-8") as fr, open(cfg.train_image_label_file, "w", encoding="utf-8") as fw:
    for line in fr:
        segs = line.strip().split(" ")
        _, fn = os.path.split(segs[0])
        fw.write(f"{fn}\t{segs[1]}\n")
