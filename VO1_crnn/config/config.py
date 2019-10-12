# -*- coding: utf-8 -*-

# @Time    : 2019/10/10
# @Author  : Lattine

# ======================
import os


class Config:
    BASE_URL = os.path.abspath(os.path.dirname(os.getcwd()))
    random_seed = 123

    # dataset
    fonts_path = os.path.join(BASE_URL, "config", "fonts")
    train_test_prefix = r"E:/tmp/zhiling"
    train_examples = 1000
    train_input_file_with_image_label = train_test_prefix + r"/train/attachments.txt"
    train_input_file_prefix = train_test_prefix + r"/train"
    train_output_data_path = train_test_prefix + r"/train_lmdb"
    test_examples = 100
    test_input_file_with_image_label = train_test_prefix + r"/test/attachments.txt"
    test_input_file_prefix = train_test_prefix + r"/test"
    test_output_data_path = train_test_prefix + r"/test_lmdb"
    img_h = 32
    img_w = 160
    alphabets = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/.%"

    # model setting
    model_name = "crnn"

    # training setting
    train_data = r"E:/tmp/zhiling/train_lmdb"
    test_data = r"E:/tmp/zhiling/test_lmdb"
    using_gpu = True
    epoches = 10000
    workers = 0
    batch_size = 64
    lr = 1e-3
    save_path = os.path.join(BASE_URL, "ckpt", model_name + f".model")
    loss_image_path = os.path.join(BASE_URL, "ckpt", model_name + f"_loss.png")

    # evalidation setting
    n_test_disp = 10
