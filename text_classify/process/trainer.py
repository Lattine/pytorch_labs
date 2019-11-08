# -*- coding: utf-8 -*-

# @Time    : 2019/10/24
# @Author  : Lattine

# ======================
import os
import torch

from data_helper import Dataset
from process import utils
from process.utils import AveragerMeter
from config import Config4TextCnn
from models import TextCNN
from config import Config4TextRnn
from models import TextRNN
from config import Config4TextRnnAtt
from models import TextRNNAtt
from config import Config4TextRCnn as Config
from models import TextRCNN as Model


class Trainer:
    def __init__(self, config, rebuild=False):
        self.config = config
        self.device = "cpu"
        self.min_loss = config.min_loss  # 用于保存最好的模型
        self.early_stop = self.config.early_stop_thresh  # 提前终止

        self._load_data(rebuild)  # 加载数据
        self.config.vocab_size = self.vocab_size
        self.nets = Model(self.config, self.word_vectors)
        utils.init_network(self.nets)
        print(self.nets)
        print("Number of Nets' Paramters: {}".format(utils.get_nets_parameters(self.nets)))

        if not rebuild:  # 尝试加载已训练的模型
            self._load_model(self.config.saved_model)

    def _load_data(self, rebuild):
        self.train_dataset = Dataset(self.config, self.device, rebuild=rebuild)
        self.train_data = self.train_dataset.gen_data(self.config.train_data)
        self.vocab_size = self.train_dataset.vocab_size
        self.word_vectors = self.train_dataset.word_vectors

    def train(self):
        optimizer = torch.optim.Adam(self.nets.parameters(), lr=self.config.lr)
        criterion = torch.nn.CrossEntropyLoss()
        loss_avg = AveragerMeter()

        for epoch in range(self.config.epoches):
            self.nets.train()
            for batch in self.train_dataset.next_batch(self.train_data, self.config.batch_size):
                x, y = batch["x"], batch["y"]
                optimizer.zero_grad()
                preds = self.nets(x)
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
        self.nets.eval()
        dataset = Dataset(self.config, self.device)
        data = dataset.gen_data(self.config.eval_data)
        n_correct = 0
        total_num = 0
        loss_avg = AveragerMeter()
        for batch in dataset.next_batch(data, self.config.batch_size):
            x, y = batch["x"], batch["y"]
            batch_size = x.size(0)
            total_num += batch_size
            with torch.no_grad():
                preds = self.nets(x)
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
            # torch.save(self.nets.state_dict(), self.config.saved_model + f"-{epoch + 1}")  # 记录历史
            torch.save(self.nets.state_dict(), self.config.saved_model)
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
            self.nets.load_state_dict(torch.load(path))
            print("Load pretrain model, and the model process will retrain.")
        else:
            print("The model will train from zero.")


if __name__ == '__main__':
    cfg = Config()
    p = Trainer(cfg, rebuild=True)
    p.train()
