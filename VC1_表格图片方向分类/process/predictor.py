# -*- coding: utf-8 -*-

# @Time    : 2019/10/17
# @Author  : Lattine

# ======================
import cv2
import albumentations as A
from albumentations.augmentations.transforms import Resize
from albumentations.pytorch import ToTensorV2
import torch
from config import Config
from models import Model


class Predictor:
    def __init__(self, config):
        self.config = config
        self.net = Model()
        self._load_model(self.config.save_ckpt_file)
        self.ix2t = {
            0: "left",
            1: "right",
            2: "up",
        }

    def predict(self, src):
        img = self._transform(self.config.img_h, self.config.img_w)(image=src)["image"]
        img = img.unsqueeze(0)

        self.net.eval()
        with torch.no_grad():
            preds = self.net(img)
        top_val, top_ix = preds.max(-1)
        return self.ix2t[top_ix.item()]

    def _transform(self, h, w):
        return A.Compose([
            Resize(h, w),
            A.Normalize(),
            ToTensorV2(),
        ])

    def _load_model(self, path):
        try:
            self.net.load_state_dict(torch.load(path))
            print(f"Successful load model from {path}")
        except Exception as e:
            print(e)
            print(f"Failed load model from {path}")


if __name__ == '__main__':
    config = Config()
    p = Predictor(config)
    src = cv2.imread(r"E:/tmp/table_classify/train/0/0c5a3e95-43f2-4631-a832-f31e4e6fd59a.png")
    p.predict(src)
