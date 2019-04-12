# -*- coding: UTF-8 -*-

# 两种主要的迁移学习的使用场景:
# 1.微调卷积网络: 取代随机初始化网络, 我们从一个预训练的网络初始化, 比如从 imagenet 1000 数据集预训练的网络. 其余的训练就像往常一样.
# 2.卷积网络作为固定的特征提取器: 在这里, 我们固定网络中的所有权重, 最后的全连接层除外. 最后的全连接层被新的随机权重替换, 并且, 只有这一层是被训练的.

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()  # interactive mode

# 1.加载数据
# 用 torchvision 和 torch.utils.data 包加载数据
# 训练一个可以区分 ants (蚂蚁) 和 bees (蜜蜂) 的模型. 用于训练的 ants 和 bees 图片各120张. 每一类用于验证的图片各75张.
# 通常, 如果从头开始训练, 这个非常小的数据集不足以进行泛化. 但是, 因为我们使用迁移学习, 应该可以取得很好的泛化效果.
# https://download.pytorch.org/tutorial/hymenoptera_data.zip
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = "hymenoptera_data"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val"]}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0) for x in ["train", "val"]}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes


# 显示一些图片
def imshow(inp, title=None):
    """Imshow for Tensor"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)


inputs, classes = next(iter(dataloaders['train']))  # 获得一批训练数据
out = torchvision.utils.make_grid(inputs)  # 从这批数据生成一个方格


# imshow(out, title=[class_names[x] for x in classes])


# 训练模型
# 通用的函数来训练模型(调度学习率, 保存最佳的学习模型)
# scheduler 参数是 torch.optim.lr_scheduler 中的 LR scheduler 对象.
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        # 每一个迭代都有训练和验证阶段
        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_correct = 0.0

            # 遍历数据
            for data in dataloaders[phase]:
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                optimizer.zero_grad()  # 清零梯度
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 深拷贝 Model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


# 显示模型的预测结果
# 一个处理少量图片, 并显示预测结果的通用函数
def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()
    for i, data in enumerate(dataloaders["val"]):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 2, 2, images_so_far)
            ax.axis("off")
            ax.set_title("predicted: {}".format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return


# 调整卷积网络
# 加载一个预训练的网络, 并重置最后一个全连接层.
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

if torch.cuda.is_available():
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # 每 7 个迭代, 让 LR 衰减 0.1 因素

# 训练和评估
# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
# visualize_model(model_ft)


# 卷积网络作为固定的特征提取器
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

if torch.cuda.is_available():
    model_conv = model_conv.cuda()

criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)  # 只有最后一层的参数被优化.
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# 训练和评估
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)
visualize_model(model_conv)

plt.ioff()
plt.show()