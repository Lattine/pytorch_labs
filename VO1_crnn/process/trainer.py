# -*- coding: utf-8 -*-

# @Time    : 2019/10/10
# @Author  : Lattine

# ======================
import os
import random

import torch
from torch.nn import CTCLoss
import numpy as np

from config import Config
from model import Model
from data_helper import dataset, utils


class Trainer:
    def __init__(self, config, retrain=False):
        self.cfg = config

        self.device = self._check_device()  # 测试是否启用GPU

        # 可重现
        random.seed(self.cfg.random_seed)
        np.random.seed(self.cfg.random_seed)
        torch.manual_seed(self.cfg.random_seed)

        self.cvt = utils.StrLabelConverter(self.cfg.alphabets)
        n_classes = len(self.cfg.alphabets) + 1

        self.net = Model(n_classes)
        self.net.apply(Trainer.init_weights)
        if not retrain: self._load_model(self.cfg.save_path)  # 加载已有模型
        self.net = self.net.to(self.device)

        # print(self.net)
        print(self._get_net_parameters())
        print(f"total classes: {n_classes}")

    def train(self):
        train_dataset = dataset.LmdbDataset(root=self.cfg.train_data, size=(self.cfg.img_w, self.cfg.img_h))
        test_dataset = dataset.LmdbDataset(root=self.cfg.test_data, transform=dataset.ResizeNormalize((self.cfg.img_w, self.cfg.img_h)))
        assert train_dataset, "process need train dataset"
        assert test_dataset, "process need test dataset"
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.workers, collate_fn=dataset.AlignCollate())
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.workers, collate_fn=dataset.AlignCollate())

        criterion = CTCLoss()
        criterion = criterion.to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

        train_losses = []
        test_losses = []
        loss_avg = utils.averager()

        for epoch in range(self.cfg.epoches):
            self.net.train()

            for train_data in iter(train_loader):
                cost = self._train_batch(criterion, optimizer, train_data)
                loss_avg.add(cost)

            # 记录损失
            train_losses.append(loss_avg.val())
            print(f"{epoch}/{self.cfg.epoches}, loss average: {loss_avg.val()}")
            loss_avg.reset()

            # 验证
            test_loss_avg = self.validation(test_loader, criterion)
            test_losses.append(test_loss_avg.val())

            # 保存模型
            torch.save(self.net.state_dict(), self.cfg.save_path)

            utils.draw_loss_plot(train_losses, test_losses, os.path.join(self.cfg.loss_image_path))  # 画出Loss曲线

    def _train_batch(self, criterion, optimizer, train_data):
        images, labels, lengths, _ = train_data
        images = images.to(self.device)
        images.requires_grad_()
        batch_size = images.size(0)

        preds = self.net(images)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        cost = criterion(preds, labels, preds_size, lengths)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        return cost

    def validation(self, test_loader, criterion, max_iter=100):
        self.net.eval()
        val_iter = iter(test_loader)
        n_correct = 0
        loss_avg = utils.averager()
        max_iter = min(max_iter, len(test_loader))
        for _ in range(max_iter):
            images, labels, lengths, texts = val_iter.next()
            images = images.to(self.device)
            batch_size = images.size(0)
            with torch.no_grad():
                preds = self.net(images)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            cost = criterion(preds, labels, preds_size, lengths)
            loss_avg.add(cost)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.cvt.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, texts):
                if pred == target.lower():
                    n_correct += 1
        raw_preds = self.cvt.decode(preds.data, preds_size.data, raw=True)[:self.cfg.n_test_disp]
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, texts):
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
        accuracy = n_correct / (max_iter * self.cfg.batch_size)
        print('Test loss: %f, accuray: %.2f%%' % (loss_avg.val(), accuracy * 100))
        return loss_avg

    # --------------------- 初始化参数 --------------------
    @staticmethod
    def init_weights(net):
        class_name = net.__class__.__name__
        if class_name.find('Conv') != -1:
            net.weight.data.normal_(0.0, 0.02)
        elif class_name.find('BatchNorm') != -1:
            net.weight.data.normal_(1.0, 0.02)
            net.bias.data.fill_(0)

    # --------------------- 模型通用方法 --------------------
    def _load_model(self, path):
        if os.path.exists(path):
            self.net.load_state_dict(torch.load(path))
            print("Load pretrain model.")
        else:
            print("The model process will retrain.")

    def _get_net_parameters(self):
        total_num = sum(p.numel() for p in self.net.parameters())
        trainable_num = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def _check_device(self):
        if not self.cfg.using_gpu:
            print("Using GPU not allow in config file!")
            return "cpu"
        if not torch.cuda.is_available():
            print("GPU is not available, process will using CPU.")
            return "cpu"
        return "cuda"


if __name__ == '__main__':
    config = Config()
    p = Trainer(config)
    p.train()
