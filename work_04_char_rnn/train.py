# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable

from dataset import CharDataset
from dataset import TextConverter
from config import cfg
from models.char_rnn import CharRNN
from visual import Visualizer


class Trainer:
    def __init__(self, converter):
        self.cvt = converter  # 文本转换器
        self.dataset = CharDataset(cfg.txt_path, cfg.seq_len, self.cvt.text2arr)  # 数据集
        self.dataloader = data.DataLoader(self.dataset, batch_size=cfg.batch_size, shuffle=True)  # 数据集包装器
        self.model = CharRNN(self.cvt.vocab_size, embedding_dim=cfg.embedding_dim, hidden_dim=cfg.hidden_dim)  # CharRNN模型

        self.criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
        self.optimizer = optim.Adam(self.model.parameters(), cfg.learning_rate)  # 优化器

    def train(self, save_every_epoch=1, plot_every=100):
        self.model.train()  # 训练模式
        if torch.cuda.is_available():
            self.model.cuda()

        # vis = Visualizer(env=cfg.env)  # 没有开启visdom服务会报错，可注释掉

        for epoch in range(cfg.epochs):  # 训练轮次
            for i, data in enumerate(self.dataloader):  # 生成器遍历训练数据
                if torch.cuda.is_available():
                    data = Variable(data.cuda().long().transpose(1, 0).contiguous())  # 包装data, 并转换维度
                else:
                    data = Variable(data.long().transpose(1, 0).contiguous())  # 包装data, 并转换维度
                x, y = data[:-1, :], data[1:, :]  # 构造input, target
                output, _ = self.model(x)  # 训练模型
                loss = self.criterion(output, y.view(-1))  # 计算损失
                self.optimizer.zero_grad()  # 梯度清零
                loss.backward()  # 反向求导
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip_norm)  # 梯度裁剪
                self.optimizer.step()  # 更新参数

                if i % plot_every == 0:
                    # vis.plot('loss', loss.item())  # 没有开启visdom服务会报错，可注释掉
                    print(loss.item())

            if epoch % save_every_epoch == 0:
                self.save_model()  # 每save_every_epoch词，保存一次模型

    def generate(self, begin, gen_length, artistic=None):
        result = []
        self.model.eval()  # 切换为测试模式
        self.load_model()  # 加载模型

        begin = self.cvt.text2arr(begin)  # 开头文本转为Index向量
        if artistic is None:  # 是否有指定的意境
            artistic = begin  # 如果没有，则使用开头作为意境
        else:
            artistic = self.cvt.text2arr(artistic)  # 意境文本转为Index向量

        hidden = None  # 开始为None，模型会随机初始化

        # 预热
        for i in range(len(artistic)):  # 遍历意境向量
            input = Variable(torch.Tensor([artistic[i]]).view(1, 1).long())  # 取1个字作为(1,1)的张量输入模型
            _, hidden = self.model(input, hidden)  # 保留hidden，抛弃output

        # 正式生成
        for i in range(gen_length):  # 生成文本的长度
            if i < len(begin):  # 遍历开头向量
                input = Variable(torch.Tensor([begin[i]]).view(1, 1).long())  # 取1个字作为(1,1)的张量输入模型
                output, hidden = self.model(input, hidden)  # 保留hidden，output(用于生成下个字符的输入)
                result.append(begin[i])  # 对于开头，直接使用给定的文本
            else:
                top_index = output.data[0].topk(1)[1][0]  # 选取模型输出Variable的Tensor中最大的坐标
                result.append(int(top_index))  # 将最大坐标加入结果，注意top_index是个引用
                input = Variable(torch.Tensor([top_index]).view(1, 1).long())  # 取当前最大坐标作为输入
                output, hidden = self.model(input, hidden)  # 生成

        return self.cvt.arr2text(result)  # 返回生成结果

    def save_model(self, type=".pth"):
        torch.save(self.model.state_dict(), cfg.saver_path + self.model.name + type)  # 保存模型

    def load_model(self, type=".pth"):
        self.model.load_state_dict(torch.load(cfg.saver_path + self.model.name + type))  # 加载模型


def train(**kwargs):
    converter = TextConverter(cfg.txt_path)
    trainer = Trainer(converter)
    trainer.train(plot_every=cfg.plot_every)


def generate(**kwargs):
    converter = TextConverter(cfg.txt_path)
    predictor = Trainer(converter)
    text = predictor.generate(cfg.start_words, cfg.gen_length)
    print("".join(text))


if __name__ == '__main__':
    train()
    generate()
