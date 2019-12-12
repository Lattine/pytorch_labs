# -*- coding: utf-8 -*-

# @Time    : 2019/12/10
# @Author  : Lattine

# ======================
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from data_helper import dataset
from data_helper import StrLabelConverter
from models import crnn as nets
from process import utils

torch.manual_seed(123)  # 可重现


class Trainer:
    def __init__(self):
        self.config = nets.Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.min_loss = self.config.min_loss  # 用于测试最好的模型
        self.early_stop = self.config.early_stop_thresh  # 提前终止

        self.cvt = StrLabelConverter(self.config.alphabets, ignore_case=True)
        n_classes = len(self.config.alphabets) + 1
        self.model = nets.Model(n_classes)
        self.model.to(self.device)

    def train(self):
        train_dataset = dataset.LmdbDataset(root=self.config.train_lmdb, height=self.config.height)
        eval_dataset = dataset.LmdbDataset(root=self.config.eval_lmdb, height=self.config.height)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.workers, collate_fn=dataset.AlignCollate())
        eval_loader = DataLoader(eval_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.workers, collate_fn=dataset.AlignCollate())

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        criterion = torch.nn.CTCLoss()
        criterion = criterion.to(self.device)
        loss_avg = utils.AveragerMeter()

        for epoch in range(self.config.epoches):
            self.model.train()

            for images, labels, lengths, texts in iter(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                images.requires_grad_()
                batch_size = images.size(0)
                preds = self.model(images)
                preds_size = torch.IntTensor([preds.size(0)] * batch_size)
                train_loss = criterion(preds, labels, preds_size, lengths)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                loss_avg.add(train_loss)

            # 记录损失
            print(f"{epoch + 1}/{self.config.epoches}, Train loss average: {loss_avg.val()}")
            loss_avg.reset()

            # 验证
            eval_loss = self._evaluation(criterion, eval_loader)
            print(f"{epoch + 1}/{self.config.epoches}, Test loss average: {eval_loss}")

            # 保存最优模型
            is_best = self._best_model(epoch, eval_loss)
            if self._early_stop(is_best):  # 验证早停
                print(f"Early stop at : {epoch - self.config.early_stop_thresh + 1}, Min Loss: {self.min_loss}")
                break

    def _evaluation(self, criterion, eval_loader):
        self.model.eval()
        n_correct = 0
        loss_avg = utils.AveragerMeter()
        max_iter = len(eval_loader)
        for images, labels, lengths, texts in iter(eval_loader):
            images = images.to(self.device)
            batch_size = images.size(0)
            with torch.no_grad():
                preds = self.model(images)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            cost = criterion(preds, labels, preds_size, lengths)
            loss_avg.add(cost)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.cvt.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, texts):
                if pred == target.lower():
                    n_correct += 1
        raw_preds = self.cvt.decode(preds.data, preds_size.data, raw=True)[:20]
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, texts):
            print(f"%-20s => %-20s, gt: %-20s" % (raw_pred, pred, gt))
        accuracy = n_correct / (max_iter * self.config.batch_size)
        print(f"Test loss: %f, accuray: %.2f%%" % (loss_avg.val(), accuracy * 100))
        return loss_avg.val()

    def _best_model(self, epoch, loss):
        if epoch == 0:
            self.min_loss = loss
            return False
        if self.min_loss > loss:
            self.min_loss = loss
            utils.check_top5(self.config.save_path)
            torch.save(self.model.state_dict(), self.config.saved_model)  # 记录历史
            return True

    def _early_stop(self, is_best):
        if not is_best:
            self.early_stop -= 1
        else:
            self.early_stop = self.config.early_stop_thresh

        if self.early_stop < 1:
            return True
        else:
            return False

    def _load_model(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            print("Load pretrain model, and the model process will retrain.")
        else:
            print("The model will train from zero.")


if __name__ == '__main__':
    p = Trainer()
    p.train()
