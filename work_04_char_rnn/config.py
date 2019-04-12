# -*- coding: UTF-8 -*-


class DefaultConfig:
    txt_path = './data/poem.txt'  # 文本文件地址

    # train param
    epochs = 30  # 训练轮数
    batch_size = 128
    seq_len = 32  # 序列长度
    embedding_dim = 128  # 词向量维度
    hidden_dim = 256  # LSTM隐藏层维度
    learning_rate = 1e-3  # 学习率
    clip_norm = 5  # 梯度裁剪
    saver_path = './ckpt/'  # 模型保存文件夹

    # generate param
    artistic_words = "细雨鱼儿出，微风燕子斜。"  # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words = "微风燕子斜，"  # 诗歌的开始
    gen_length = 48  # 生成字数

    # visual param
    env = "poertry"  # visdom名称
    plot_every = 50  # 记录频率


cfg = DefaultConfig()
