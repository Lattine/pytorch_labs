# -*- coding: utf-8 -*-

# @Time    : 2019/10/24
# @Author  : Lattine

# ======================
import os


class Config(object):
    BASE_URL = os.path.abspath(os.path.dirname(os.getcwd()))
    rebuild = False  # 是否重新构建模型

    # ---------- 数据集 ----------
    train_data = os.path.join(BASE_URL, "data", "train.csv")  # 训练数据
    eval_data = os.path.join(BASE_URL, "data", "eval.csv")  # 验证数据
    test_data = os.path.join(BASE_URL, "data", "test.csv")  # 预测数据
    stopwords = None  # os.path.join(BASE_URL, "data", "stopwords.txt")  # 停用词文件
    word_vectors = os.path.join(BASE_URL, "data", "sgns.weibo.char")  # 词向量文件
    output_path = os.path.join(BASE_URL, "data", "output")  # 字典等数据集输出目录
    embedding_size = 300
    vocab_size = 10001
    sequence_length = 126  # 需要看数据集，具体调试

    # ---------- 模型 -----------
    model_name = "text_cnn"
    save_path = os.path.join(BASE_URL, "ckpt")
    saved_model = os.path.join(save_path, model_name + ".pth")
    num_classes = 2
    embedding_size = 300
    filter_sizes = (2, 3, 4)  # 卷积核尺寸
    num_filters = 256
    dropout = 0.5

    # ---------- Train ----------
    lr = 1e-3
    batch_size = 64
    epoches = 100
    min_loss = 1000.0
    early_stop_thresh = 10  # 提前终止的轮数

    # ---------- Train ----------
    predict_result = os.path.join(BASE_URL, "data", "result.csv")


# 以下用于自动创建所需目录
config = Config()
if not os.path.exists(config.save_path):
    os.makedirs(config.save_path)
