# -*- coding: utf-8 -*-

# @Time    : 2019/10/24
# @Author  : Lattine

# ======================
import os
import torch
from data_helper import Dataset
from process import utils
from models import text_cnn as nets


class Trainer:
    def __init__(self, rebuild=False):
        self.config = nets.Config()
        self.config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config.rebuild = rebuild

        self.min_loss = self.config.min_loss  # 用于测试最好的模型
        self.early_stop = self.config.early_stop_thresh  # 提前终止

        self._load_data()  # 加载数据，以确定词袋大小
        self.config.vocab_size = self.vocab_size
        self.model = nets.Model(self.config, self.word_vectors)

        if not rebuild:  # 如果不是重构模型，尝试加载已训练的模型
            self._load_model(self.config.saved_model)

        self.model.to(self.config.device)

    def _load_data(self):
        train_data = Dataset(self.config, force_build=self.config.rebuild)
        self.train_iter = train_data.dataset_iter
        self.vocab_size = train_data.vocab_size
        self.word_vectors = train_data.word_vectors
        self.eval_iter = Dataset(self.config, type="eval", force_build=False).dataset_iter

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        criterion = torch.nn.CrossEntropyLoss()
        loss_avg = utils.AveragerMeter()

        for epoch in range(self.config.epoches):
            self.model.train()
            for x, y in self.train_iter:
                optimizer.zero_grad()
                preds = self.model(x)
                train_loss = criterion(preds, y.view(-1))
                loss_avg.add(train_loss)
                train_loss.backward()
                optimizer.step()
            print(f"Epoch: {epoch + 1}, Train loss: {loss_avg.val()}")
            loss_avg.reset()

            eval_loss, eval_acc = self._evaluation(criterion)
            print(f"Epoch: {epoch + 1}, Test loss: {eval_loss}, Accuray: {eval_acc * 100}%")

            is_best = self._best_model(epoch, eval_loss)

            if self._early_stop(is_best):  # 验证早停
                print(f"Early stop at : {epoch + 1}, Min Loss: {self.min_loss}")
                break

    def _evaluation(self, criterion):
        self.model.eval()
        n_correct = 0
        total_num = 0
        loss_avg = utils.AveragerMeter()
        for x, y in self.eval_iter:
            batch_size = x.size(0)
            total_num += batch_size
            with torch.no_grad():
                preds = self.model(x)
            labels = y.squeeze()
            loss = criterion(preds, labels)
            loss_avg.add(loss)

            top_val, top_ix = preds.max(-1)
            assert len(top_ix) == len(labels)
            for p, t in zip(top_ix, labels):
                if p == t:
                    n_correct += 1
        accuracy = n_correct / total_num
        return loss_avg.val(), accuracy

    def _best_model(self, epoch, loss):
        if epoch == 0:
            self.min_loss = loss
            return False
        if self.min_loss > loss:
            self.min_loss = loss
            utils.check_top5(self.config.save_path)
            torch.save(self.model.state_dict(), self.config.saved_model)
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
    p = Trainer(rebuild=True)
    p.train()
