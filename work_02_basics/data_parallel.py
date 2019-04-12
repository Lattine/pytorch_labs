# -*- coding: UTF-8 -*-

import torch
import torch.nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# 参数和数据加载
input_size = 5
output_size = 2
batch_size = 30  # 多GPU分片Batch_Size
data_size = 100


# 伪数据集
class RandomDataSet(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


rand_loader = DataLoader(dataset=RandomDataSet(input_size, 100), batch_size=batch_size, shuffle=True)

# 简单模型
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("In Model: input size", input.size(), "output size", output.size())
        return output


# 创建模型和DataParallel
# 首先, 我们需要生成一个模型的实例并且检测我们是否拥有多个 GPU.如果有多个GPU ,
# 我们可以使用 nn.DataParallel 来包装我们的模型,
# 然后我们就可以将我们的模型通过 model.gpu() 施加于这些GPU上
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
if torch.cuda.is_available():
    model.cuda()

# 运行模型
for data in rand_loader:
    if torch.cuda.is_available():
        input_var = Variable(data.cuda())
    else:
        input_var = Variable(data)
    output = model(input_var)
    print("Outside: input size", input_var.size(), "output_size", output.size())

# DataParallel 自动地将数据分割并且将任务送入多个GPU上的多个模型中进行处理.
# 在每个模型完成任务后, DataParallel 采集和合并所有结果, 并将最后的结果呈现给你.
