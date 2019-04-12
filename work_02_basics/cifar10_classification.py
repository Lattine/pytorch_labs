# -*- coding: UTF-8 -*-

import torch
import torchvision
import torchvision.transforms as tfs

# 1.加载并规范化数据集
# torchvision 数据集的输出是范围 [0, 1] 的 PILImage 图像. 将它们转换为归一化范围是[-1,1]的张量
transforms = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2.定义一个卷积神经网络
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Cifar10Net(nn.Module):
    def __init__(self):
        super(Cifar10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # [-1,3,32,32] => [-1,6,28,28]
        self.pool1 = nn.MaxPool2d(2, 2)  # [-1,6,28,28] => [-1,6,14,14]
        self.conv2 = nn.Conv2d(6, 16, 5)  # [-1,6,14,14] => [-1,16,10,10]
        self.pool2 = nn.MaxPool2d(2, 2)  # [-1,16,10,10] => [-1,16,5,5]
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Cifar10Net()

# 3.定义一个损失函数和优化器
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

# 4.训练网络, 只需循环遍历数据迭代器, 并将输入提供给网络和优化器.
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data  # 得到输入数据
        inputs, labels = Variable(inputs), Variable(labels)  # 包装数据
        optimizer.zero_grad()  # 清零梯度
        # forward + backward + optimize
        output = net(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # 打印日志
        running_loss += loss.data[0]
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print("Finished Training")

# 5.在测试数据上测试网络
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# 看看哪些类别表现良好, 哪些类别表现不佳
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(10):
    # print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    pass