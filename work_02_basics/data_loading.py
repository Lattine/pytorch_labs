# -*- coding: UTF-8 -*-

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings

warnings.filterwarnings("ignore")
plt.ion()

# 读取CSV文件, 以(N,2)的数组形式获得标记点, 其中N表示标记点的个数.
landmarks_frame = pd.read_csv("faces/face_landmarks.csv")

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype("float").reshape(-1, 2)

print("Image name: {}".format(img_name))
print("Landmarks shape: {}".format(landmarks.shape))
print("First 4 Landmarks: {}".format(landmarks[:4]))


# 写一个函数来显示一张图片和它的标记点, 然后用这个函数来显示一个样本.
def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker=".", c="r")
    plt.pause(3)


# plt.figure()
# show_landmarks(io.imread(os.path.join("faces/", img_name)), landmarks)
# plt.show()


# Dataset类
# 自己的数据集一般应该继承``Dataset``, 并且重写下面的方法:
# __len__ 使用``len(dataset)`` 可以返回数据集的大小
# __getitem__ 支持索引, 以便于使用 dataset[i] 可以 获取第:math:i个样本(0索引)
class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype("float").reshape(-1, 2)
        sample = {"image": image, "landmarks": landmarks}

        if self.transform:
            sample = self.transform(sample)
        return sample


# 实例化这个类 并且迭代所有的数据样本. 我们将打印前4个样本, 并显示它们的标记点.
face_dataset = FaceLandmarksDataset(csv_file="faces/face_landmarks.csv", root_dir="faces/")

fig = plt.figure()
for i in range(len(face_dataset)):
    sample = face_dataset[i]
    print(i, sample["image"].shape, sample["landmarks"].shape)
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title("sample #{}".format(i))
    ax.axis("off")
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break


# 可以看到上面输出的样例中的图像并不是同一尺寸的图片. 大多数神经网络需要输入 一个固定大小的图像, 因此我们需要写代码来处理. 让我们创建三个transform操作:
# Rescale: 修改图片尺寸
# RandomCrop: 随机裁切图片, 这是数据增强的方法
# ToTensor: 将numpy格式的图片转为torch格式的图片（我们需要交换坐标轴）
class Rescale:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / new_h]
        return {"image": img, "landmarks": landmarks}


class RandomCrop:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        img = image[top: top + new_h, left:left + new_w]
        landmarks = landmarks - [left, top]
        return {"image": img, "landmarks": landmarks}


class ToTensor:
    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        # 交换颜色通道, 因为
        # numpy图片: H x W x C
        # torch图片: C X H X W
        image = image.transpose((2, 0, 1))
        return {"image": torch.from_numpy(image), "landmarks": torch.from_numpy(landmarks)}


scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256), RandomCrop(224)])

fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)
    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)
plt.show()

# 迭代整个数据集
# 用简单的``for``循环来迭代整个数据集会丢失很多特点, 特别地, 我们会丢失:
# 批读取数据
# 打乱数据顺序
# 使用``multiprocessing``并行加载数据
transformed_dataset = FaceLandmarksDataset(csv_file="faces/face_landmarks.csv", root_dir="faces/", transform=transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))
for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]
    print(i, sample["image"].size(), sample["landmarks"].size())
    if i == 3:
        break

dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=0)


# 定义一个函数来查看某个 batch 的数据样本图片和标记点
def show_landmarks_batch(sample_batched):
    images_batch, landmarks_batch = sample_batched["image"], sample_batched["landmarks"]
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    for i in range(batch_size):
        for i in range(batch_size):
            plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size, landmarks_batch[i, :, 1].numpy(), s=10, marker=".", c="r")
            plt.title("Batch from dataloader")


for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched["image"].size(), sample_batched["landmarks"].size())

    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis("off")
        plt.ioff()
        break
