# -*- coding: utf-8 -*-

# @Time    : 2019/10/16
# @Author  : Lattine

# ======================
import albumentations as A
from albumentations.augmentations.transforms import Resize
from albumentations.pytorch import ToTensorV2


def train_transform(h, w):
    return A.Compose([
        Resize(h, w),
        A.Normalize(),
        ToTensorV2(),
    ])
