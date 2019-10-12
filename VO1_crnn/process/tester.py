# -*- coding: utf-8 -*-

# @Time    : 2019/10/12
# @Author  : Lattine

# ======================
import os
import time
import argparse

import torch
from PIL import Image

from config import Config
from model import Model
from data_helper import utils, dataset


class Tester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cvt = utils.StrLabelConverter(self.cfg.alphabets)
        n_classes = len(self.cfg.alphabets) + 1
        self.net = Model(n_classes)
        self._load_model(self.cfg.save_path)

    def test(self, img_path):
        start = time.time()
        image = Image.open(img_path).convert('L')
        w, h = image.size
        transformer = dataset.ResizeNormalize((int(w / h * 32), 32))
        image = transformer(image)
        image = image.view(1, *image.size())

        self.net.eval()
        with torch.no_grad():
            preds = self.net(image)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = torch.IntTensor([preds.size(0)])
        raw_pred = self.cvt.decode(preds.data, preds_size.data, raw=True)
        sim_pred = self.cvt.decode(preds.data, preds_size.data, raw=False)
        print('%-20s => %-20s' % (raw_pred, sim_pred))
        print("Cost %.2f s" % (time.time() - start))

    def _load_model(self, path):
        if os.path.exists(path):
            self.net.load_state_dict(torch.load(path))
            print("! load pretrain model.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='../data/demo.png', help='The name of image to be tested.')
    args = parser.parse_args()
    config = Config()
    p = Tester(config)
    p.test(args.img)
