# -*- coding: utf-8 -*-

# @Time    : 2019/10/24
# @Author  : Lattine

# ======================
import os


class Config(object):
    BASE_URL = os.path.abspath(os.path.dirname(os.getcwd()))

    # ---------- 数据集 ----------
    train_data = os.path.join(BASE_URL, "data", "train.csv")  # 训练数据
    eval_data = os.path.join(BASE_URL, "data", "eval.csv")  # 验证数据
    test_data = os.path.join(BASE_URL, "data", "test.csv")  # 预测数据
    stopwords = None  # os.path.join(BASE_URL, "data", "stopwords.txt")  # 停用词文件
    word_vectors = os.path.join(BASE_URL, "data", "sgns.weibo.char")  # 词向量文件
    output_path = os.path.join(BASE_URL, "data", "output")  # 字典等数据集输出目录
    embedding_size = 300  # 词向量维度
    vocab_size = 100001  # 字典大小，字=1w，词=10w+

    # ---------- Train ----------
    save_path = os.path.join(BASE_URL, "ckpt")
    lr = 1e-3
    batch_size = 64
    epoches = 100
    min_loss = 1000.0
    early_stop_thresh = 10  # 提前终止的轮数

    # ---------- Predict ----------
    predict_result = os.path.join(BASE_URL, "data", "result.csv")


class Config4TextCnn(Config):
    sequence_length = 100  # 每条文本截取的长度

    # ---------- 模型 -----------
    model_name = "text_cnn"
    saved_model = os.path.join(Config.save_path, model_name + ".pth")
    num_classes = 2
    filter_sizes = (2, 3, 4)  # 卷积核尺寸
    num_filters = 256
    dropout = 0.5


class Config4TextRnn(Config):
    sequence_length = 60  # 每条文本截取的长度

    # ---------- 模型 -----------
    model_name = "text_rnn"
    saved_model = os.path.join(Config.save_path, model_name + ".pth")
    num_classes = 2
    hidden_size = 128
    num_layers = 2
    dropout = 0.5


class Config4TextRnnAtt(Config):
    sequence_length = 32  # 每条文本截取的长度

    # ---------- 模型 -----------
    model_name = "text_rnn_att"
    saved_model = os.path.join(Config.save_path, model_name + ".pth")
    num_classes = 2
    rnn_hidden_size = 128
    att_hidden_size = 64
    num_layers = 2
    dropout = 0.5


class Config4TextRCnn(Config):
    sequence_length = 64  # 每条文本截取的长度

    # ---------- 模型 -----------
    model_name = "text_rcnn"
    saved_model = os.path.join(Config.save_path, model_name + ".pth")
    num_classes = 2
    rnn_hidden_size = 256
    num_layers = 1
    dropout = 0.5


# 以下用于自动创建所需目录
config = Config()
if not os.path.exists(config.save_path):
    os.makedirs(config.save_path)
