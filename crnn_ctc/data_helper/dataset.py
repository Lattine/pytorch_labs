# -*- coding: utf-8 -*-

# @Time    : 2019/12/10
# @Author  : Lattine

# ======================
import six
import sys
import cv2
import lmdb
import torch
import numpy as np
from torch.utils.data import Dataset


class LmdbDataset(Dataset):
    def __init__(self, root, transform=None, height=32):
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print(f"cannot create lmdb form {root}")
            sys.exit(0)
        with self.env.begin(write=False) as txn:
            self.n_samples = int(txn.get("total-samples".encode()))
        self.transform = ResizeNormalize(height) if transform is None else transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, ix):
        assert ix < len(self), "INDEX ERROR"
        with self.env.begin(write=False) as txn:
            image_key = "image-%09d" % ix
            text_key = 'text-%09d' % ix
            label_key = "label-%09d" % ix
            length_key = 'length-%09d' % ix
            try:
                img_buf = txn.get(image_key.encode())
                image_buf = np.fromstring(img_buf, dtype=np.uint8)
                img = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
                img = self.transform(img)
                img = img.unsqueeze(0)
            except IOError:
                print(f"raise error for {image_key}")
                return self[ix + 1]

            label = eval(txn.get(label_key.encode()).decode())
            length = eval(txn.get(length_key.encode()).decode())
            label, length = torch.IntTensor(label), torch.IntTensor(length)
            text = txn.get(text_key.encode()).decode()

        return img, label, length, text


class ResizeNormalize:
    def __init__(self, height):
        self.h = height

    def __call__(self, src):
        img = (255 - src) / 255.0  # 归一化
        img = torch.FloatTensor(img)
        return img


class AlignCollate:
    def __init__(self):
        pass

    def __call__(self, batch):
        images, labels, lengths, text = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        lengths = torch.cat(lengths)

        return images, labels, lengths, text
