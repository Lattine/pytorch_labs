# -*- coding: utf-8 -*-

# @Time    : 2019/10/15
# @Author  : Lattine

# ======================
import sys

import cv2
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from .trans import train_transform
from config import Config


class CustomDataset(Dataset):
    def __init__(self, lmdb_root, is_train=True):
        self.env = lmdb.open(lmdb_root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print(f"cannot create lmdb form {lmdb_root}")
            sys.exit(0)
        with self.env.begin(write=False) as txn:
            self.n_samples = int(txn.get("n_samples".encode()))
        self.is_train = is_train

    def __len__(self):
        return self.n_samples

    def __getitem__(self, ix):
        assert ix <= len(self), "index out of range!"
        ix += 1  # 在存储时，图片key从 1 开始计数
        with self.env.begin(write=False) as txn:
            try:
                image_key = "image-%09d" % ix
                label_key = "label-%09d" % ix
                image_bin = txn.get(image_key.encode())
                image_buf = np.fromstring(image_bin, dtype=np.uint8)
                img = cv2.imdecode(image_buf, cv2.IMREAD_ANYCOLOR)
                if self.is_train:
                    img = train_transform(Config.img_h, Config.img_w)(image=img)["image"]

                label = int(txn.get(label_key.encode()).decode())
                label = torch.LongTensor([label])
            except:
                return self[ix + 1]
        return img, label
