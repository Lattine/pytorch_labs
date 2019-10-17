# -*- coding: utf-8 -*-

# @Time    : 2019/10/16
# @Author  : Lattine

# ======================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config
from data_helper import CustomDataset
from utils import AveragerMeter
from models import Model


class Trainer:
    def __init__(self, config):
        self.config = config
        self.net = Model()
        self.writer = SummaryWriter(self.config.logs_path)

    def train(self):
        train_data = CustomDataset(config.train_data_lmdb, is_train=True)
        test_data = CustomDataset(config.test_data_lmdb, is_train=True)
        train_data_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        test_data_loader = DataLoader(test_data, batch_size=self.config.batch_size, shuffle=True)

        optimizer = optim.Adam(self.net.parameters(), self.config.lr)
        criterion = nn.CrossEntropyLoss()

        loss_avg = AveragerMeter()
        best_eval_loss = 100.0
        for epoch in range(self.config.epoches):
            self.net.train()
            for imgs, labs in train_data_loader:
                optimizer.zero_grad()

                preds = self.net(imgs)
                labs = labs.squeeze()  # [[0], [1]] => [0, 1]
                loss = criterion(preds, labs)
                loss_avg.add(loss)

                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}: Loss：{loss_avg.val()}")
            self.writer.add_scalar("Train Loss", loss_avg.val(), epoch + 1)
            loss_avg.reset()

            eval_loss, eval_acc = self.evaluation(test_data_loader, criterion)
            print(f"Test loss: {eval_loss.val()}, Accuray: {eval_acc * 100}%")
            self.writer.add_scalar("Test Loss", eval_loss.val(), epoch + 1)
            self.writer.add_scalar("Accuracy", eval_acc, epoch + 1)

            # 保存模型 最好的Validation模型
            if eval_loss.val() < best_eval_loss:
                best_eval_loss = eval_loss.val()
                torch.save(self.net.state_dict(), self.config.save_ckpt_file + f"-{epoch + 1}")

    def evaluation(self, data_loader, criterion, max_iter=100):
        self.net.eval()
        data_iter = iter(data_loader)
        n_correct = 0
        loss_avg = AveragerMeter()
        max_iter = min(max_iter, len(data_loader))
        for _ in range(max_iter):
            images, labels = data_iter.next()
            with torch.no_grad():
                preds = self.net(images)
            labels = labels.squeeze()
            loss = criterion(preds, labels)
            loss_avg.add(loss)

            top_val, top_ix = preds.max(-1)
            assert len(top_ix) == len(labels)
            for p, t in zip(top_ix, labels):
                if p == t:
                    n_correct += 1
        accuracy = n_correct / (max_iter * self.config.batch_size)
        return loss_avg, accuracy


if __name__ == '__main__':
    config = Config()
    p = Trainer(config)
    p.train()
