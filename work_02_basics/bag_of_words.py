# -*- coding: UTF-8 -*-

# 基于词袋表示法的逻辑斯蒂回归分类器
# 我们的模型将映射一个稀疏的BOW表示来记录标签上的概率.我们为词汇表中的每个单词分配一个索引.
# 例如, 我们的完整的词汇表有两个单词: “你好” 和 “世界”, 这两个单词的索引分别为0和1.
# 句子为 “hello hello hello hello” 的BoW向量为 [4,0]
# 对于 “hello world world hello” , 它是 [2,2]
# 等等.一般来说, 它是 [Count(hello),Count(world)]
# 将这个BOW向量表示为 x. 我们的网络输出是: logSoftmax(Ax+b)
# 也就是说, 我们通过affine map传递输入, 然后进行softmax.

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

word_to_idx = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
print(word_to_idx)
VOCAB_SIZE = len(word_to_idx)
NUM_LABELs = 2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class BowClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(BowClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.softmax(self.linear(bow_vec), dim=1)


def make_bow_vec(sentence, word_to_idx):
    vec = torch.zeros(len(word_to_idx))
    for word in sentence:
        vec[word_to_idx[word]] += 1
    return vec


def make_bow_label(label, label_to_idx):
    return torch.LongTensor([label_to_idx[label]])


model = BowClassifier(NUM_LABELs, VOCAB_SIZE)
# model知道它的系数.第一个输出的是A, 第二个是b.
# 当你在模块__init__函数中指定一个神经元去分类变量, self.linear = nn.Linear(...)被执行
# 随后从Pytorch devs通过Python magic, 你的模块(在本例中, BoWClassifier) 将会存储 nn.Linear的系数
for param in model.parameters():
    print(param)

sample = data[0]
bow_vector = make_bow_vec(sample[0], word_to_idx)
log_probs = model(autograd.Variable(bow_vector))
print(log_probs)

label_to_idx = {"SPANISH": 0, "ENGLISH": 1}

for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vec(instance, word_to_idx))
    log_probs = model(bow_vec)
    print(log_probs)
print(next(model.parameters())[:, word_to_idx["creo"]])

loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    for instance, label in data:
        model.zero_grad()

        bow_vec = autograd.Variable(make_bow_vec(instance, word_to_idx))
        target = autograd.Variable(make_bow_label(label, label_to_idx))
        log_probs = model(bow_vec)

        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_label(instance, word_to_idx))
    log_probs = model(bow_vec)
    print(log_probs)

print(next(model.parameters())[:, word_to_idx["creo"]])
