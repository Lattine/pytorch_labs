# -*- coding: utf-8 -*-

# @Time    : 2019/10/15
# @Author  : Lattine

# ======================
import os


class Config:
    BASE_URL = os.path.abspath(os.path.dirname(os.getcwd()))

    # dataset settings
    data_prefix = r"E:/tmp/table_classify/"
    train_data_path = data_prefix + r"train/"
    train_data_img_label_file = data_prefix + r"train_data_img_label_file.txt"
    train_data_lmdb = data_prefix + r"train_lmdb/"
    test_data_path = data_prefix + r"test/"
    test_data_img_label_file = data_prefix + r"test_data_img_label_file.txt"
    test_data_lmdb = data_prefix + r"test_lmdb/"
    map_size = 9511627776
    img_h = 1024
    img_w = 1024

    # model settings
    model_name = "table_classify"
    save_path = os.path.join(BASE_URL, "ckpt")
    save_ckpt_file = os.path.join(save_path, model_name+".pth")
    logs_path = os.path.join(BASE_URL, "ckpt", "logs")

    # train settings
    epoches = 10
    batch_size = 8
    lr = 1e-3


# 以下用于自动创建所需目录
config = Config()
if not os.path.exists(config.save_path):
    os.makedirs(config.save_path)
if not os.path.exists(config.logs_path):
    os.makedirs(config.logs_path)
