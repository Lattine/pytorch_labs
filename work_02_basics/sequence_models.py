# -*- coding: UTF-8 -*-

#  Pytorch 中, LSTM 的所有的形式固定为3D 的 tensor. 每个维度有固定的语义含义, 不能乱掉.
# 其中第一维是序列本身, 第二维以 mini-batch 形式 来索引实例, 而第三维则索引输入的元素.

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print(1, torch.randn(1, 3))
print(2, torch.randn((1, 3)))
print("-" * 100)
lstm = nn.LSTM(3, 3)
inputs = [autograd.Variable(torch.randn((1, 3))) for _ in range(5)]  # 构造一个长度为5的序列
hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(torch.randn((1, 1, 3))))

for i in inputs:
    # 将序列的元素逐个输入到LSTM
    # 经过每步操作,hidden 的值包含了隐藏状态的信息
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# 另外, 我们还可以一次对整个序列进行训练. LSTM 返回的第一个值表示所有时刻的隐状态值,
# 第二个值表示最近的隐状态值 (因此下面的 "out"的最后一个值和 "hidden" 的值是一样的).
# 之所以这样设计, 是为了通过 "out" 的值来获取所有的隐状态值, 而用 "hidden" 的值来
# 进行序列的反向传播运算, 具体方式就是将它作为参数传入后面的 LSTM 网络.

# 增加额外的第二个维度
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(torch.randn((1, 1, 3))))
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)

print("-" * 100)


#  用 LSTM 来进行词性标注
# 准备数据
def prepare_sequence(seq, to_ix):
    idx = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idx)
    return autograd.Variable(tensor)


training_data = [
    ("我 是 中 国 人".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("你 也 是 的".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for w in sent:
        if w not in word_to_ix:
            word_to_ix[w] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# 实际中通常使用更大的维度如32维, 64维.
# 这里我们使用小的维度, 为了方便查看训练过程中权重的变化.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


# 构造模型
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        #  LSTM 以 word_embeddings 作为输入, 输出维度为 hidden_dim 的隐状态值
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # 线性层将隐状态空间映射到标注空间
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 开始时刻, 没有隐状态
        # 关于维度设置的详情,请参考 Pytorch 文档
        # 各个维度的含义是 (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)), autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentense):
        embeds = self.word_embeddings(sentense)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentense), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentense), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# 训练模型
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 查看下训练前得分的值
# 注意: 输出的 i,j 元素的值表示单词 i 的 j 标签的得分
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)

for epoch in range(300):
    for sentence, tags in training_data:
        # Step 1. 请记住 Pytorch 会累加梯度
        # 每次训练前需要清空梯度值
        model.zero_grad()
        # 此外还需要清空 LSTM 的隐状态
        # 将其从上个实例的历史中分离出来
        model.hidden = model.init_hidden()
        # Step 2. 准备网络输入, 将其变为词索引的 Variables 类型数据
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        # Step 2. 前向传播
        tag_scores = model(sentence_in)
        # Step 4. 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
# 句子是 "the dog ate the apple", i,j 表示对于单词 i, 标签 j 的得分.
# 我们采用得分最高的标签作为预测的标签. 从下面的输出我们可以看到, 预测得
# 到的结果是0 1 2 0 1. 因为 索引是从0开始的, 因此第一个值0表示第一行的
# 最大值, 第二个值1表示第二行的最大值, 以此类推. 所以最后的结果是 DET
# NOUN VERB DET NOUN, 整个序列都是正确的!
print(tag_scores)

# 使用字符级特征来增强 LSTM 词性标注器
# 上面的例子中, 每个词都有一个词嵌入, 作为序列模型的输入. 接下来让我们使用每个的单词的 字符级别的表达来增强词嵌入.
# 我们期望这个操作对结果能有显著提升, 因为像词缀这样的字符级 信息对于词性有很大的影响. 比如说, 像包含词缀 -ly 的单词基本上都是被标注为副词.
# 具体操作如下. 用 cw 来表示单词 w 的字符级表达, 同之前一样, 我们使 用 xw 来表示词嵌入. 序列模型的输入就变成了 xw 和 cw 的拼接.
# 因此, 如果 xw 的维度是5, cw 的维度是3, 那么我们的 LSTM 网络的输入维度大小就是8.
#
# 为了得到字符级别的表达, 将单词的每个字符输入一个 LSTM 网络, 而 cw 则为这个 LSTM 网络最后的隐状态. 一些提示:
# 新模型中需要两个 LSTM, 一个跟之前一样, 用来输出词性标注的得分, 另外一个新增加的用来 获取每个单词的字符级别表达.
# 为了在字符级别上运行序列模型, 你需要用嵌入的字符来作为字符 LSTM 的输入.
