# -*- coding: utf-8 -*-

# @Time    : 2019/10/24
# @Author  : Lattine

# ======================
"""
    # 构造数据加载器逻辑
    # 1. 读取原始数据，分开文本、标签
    # 2. 过滤停用词、低频词
    # 3. 训练时，根据过滤后的训练语料，构造出索引字典；测试时，加载索引字典
    # 4. 将文本、标签转为索引表示
    # 5. 将文本截断或者PADDING成固定长度
    # 6. 文本、标签持久化，返回
"""
import os
import pickle
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm
from data_helper.dataset_read import read_file
from data_helper.utils import remove_files

EXTEND_WORDS = [("<PAD>", 0), ("<UNK>", 1)]


class DatasetBase:
    def __init__(self, config, device, rebuild=False):
        self._output_path = config.output_path  # 输出目录
        self._stopwords_path = config.stopwords if config.stopwords else None  # 停用词文件
        self._word_vectors_path = config.word_vectors if config.word_vectors else None  # 词向量文件
        self._embedding_size = config.embedding_size  # 词向量维度
        self._sequence_length = config.sequence_length  # 序列最大长度

        self.vocab_size = config.vocab_size  # 字典大小，初始化为配置中的大小，后期会根据实际自动调整
        self.word_vectors = None  # 词向量矩阵

        self.device = device

        self._check_path(self._output_path)
        if rebuild:
            remove_files(self._output_path)
            print("Last output tmp files is clear.")

    def next_batch(self, data, batch_size):
        x, y = data["x"], data["y"]
        perm = np.arange(len(x))
        np.random.shuffle(perm)
        x = x[perm]
        y = y[perm]

        num_batches = len(x) // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            xs = torch.LongTensor(x[start: end]).to(self.device)
            ys = torch.LongTensor(y[start: end]).to(self.device)

            yield dict(x=xs, y=ys)

    def _read_data(self, path):
        return read_file(path)

    def _remove_stopwords_and_low_frequent(self, inputs):
        # 如果设置停用词表，去除停用词
        if self._stopwords_path:
            with open(self._stopwords_path, "r", encoding="utf-8") as fr:
                stopwords = [line.strip() for line in fr]
                stopwords = set(stopwords)

            inputs = [[w for w in sent if w not in stopwords] for sent in inputs]

        # 统计词频，根据设置的字典大小去除停用词
        words = self._count_words(inputs, self.vocab_size - 4)
        inputs = [[w for w in sent if w in words] for sent in inputs]
        return inputs

    def _vocab(self, inputs, labels):
        w2ix_path = os.path.join(self._output_path, 'w2ix.pkl')
        t2ix_path = os.path.join(self._output_path, 't2ix.pkl')

        # ----------- 加载，或生成字典 -------------
        if os.path.exists(w2ix_path) and os.path.exists(t2ix_path):
            with open(w2ix_path, "rb") as fr:
                w2ix = pickle.load(fr)
            with open(t2ix_path, "rb") as fr:
                t2ix = pickle.load(fr)
            self.vocab_size = len(w2ix)
        else:
            words = self._count_words(inputs)
            words = ["<PAD>", "<UNK>"] + list(words)
            vocab = words[:self.vocab_size]  # 词汇表上限
            self.vocab_size = len(vocab)  # 如果vocab的长度小于config设置的值，则用实际长度

            # 将词汇-索引字典保存
            w2ix = {w: i for i, w in enumerate(vocab)}
            with open(w2ix_path, 'wb') as fw:
                pickle.dump(w2ix, fw)

            # 将标签-索引字典保存
            unique_labels = list(set(labels))
            t2ix = {w: i for i, w in enumerate(unique_labels)}
            with open(t2ix_path, "wb") as fw:
                pickle.dump(t2ix, fw)

        # 加载词向量文件
        w2v_path = os.path.join(self._output_path, "w2v.pkl")
        if os.path.exists(w2v_path):  # 加载预先存储的词向量
            with open(w2v_path, "rb") as fr:
                word_vectors = pickle.load(fr)
                self.word_vectors = torch.FloatTensor(word_vectors)
        elif self._word_vectors_path:  # 基于公开词向量，抽取语料相关的词向量
            word_vectors = self._get_word_vectors(w2ix)
            with open(w2v_path, "wb") as fw:
                pickle.dump(word_vectors, fw)
            self.word_vectors = torch.FloatTensor(word_vectors)

        return w2ix, t2ix

    def _trans2ix(self, inputs, vocab2ix):
        """数据转为索引"""
        inputs_idx = [[vocab2ix.get(w, vocab2ix.get("<UNK>")) for w in sentence] for sentence in inputs]
        return inputs_idx

    def _trans2ix_with_onehot(self, labels, t2ix):
        """标签转索引， 使用np.eye()快速生成One-Hot编码"""
        num_classes = len(t2ix)
        labels_idx = [t2ix.get(label) for label in labels]
        onehots = np.eye(num_classes)[labels_idx]
        return onehots.tolist()

    def _padding(self, inputs, poster=False):
        """ 固定长度"""
        if poster:
            inputs_new = [sent[:self._sequence_length] if len(sent) > self._sequence_length else sent + [0] * (self._sequence_length - len(sent)) for sent in inputs]
        else:
            inputs_new = [sent[:self._sequence_length] if len(sent) > self._sequence_length else [0] * (self._sequence_length - len(sent)) + sent for sent in inputs]
        return inputs_new

    def _pickle_dump(self, data, data_path):
        with open(data_path, 'wb') as fw:
            pickle.dump(data, fw)

    def _get_word_vectors(self, w2ix):
        vocab = set(w2ix.keys())
        word_vectors = (1.0 / np.sqrt(len(vocab)) * (2 * np.random.rand(len(vocab), self._embedding_size) - 1))  # (-1,1)/sqrt(n)
        with open(self._word_vectors_path, encoding='utf-8', errors='ignore') as fr:
            for line in tqdm(fr):
                try:
                    segs = line.strip().split()
                    if segs[0] in vocab:
                        word_vectors[w2ix.get(segs[0]), :] = segs[1:]
                except:
                    continue
        return word_vectors

    def _check_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def _count_words(self, inputs, size=None):
        word_counts = Counter()
        for sent in inputs:
            word_counts.update(sent)
        words = set()
        for k, v in word_counts.most_common(size):  # 统计最常用的词，为词表大小减去<START>,<UNK>,<PAD>,<END>
            words.add(k)
        return words


class Dataset(DatasetBase):
    def __init__(self, config, device, rebuild=False):
        super(Dataset, self).__init__(config, device, rebuild)

    def gen_data(self, path, filename=None):
        inputs, labels = self._read_data(path=path)  # 读入数据
        inputs = self._remove_stopwords_and_low_frequent(inputs)  # 去除停用词、低频词
        w2ix, t2ix = self._vocab(inputs, labels)  # 构造或加载词表
        inputs = self._trans2ix(inputs, w2ix)  # 文本转ID
        labels = self._trans2ix(labels, t2ix)  # 标签转ID
        inputs = self._padding(inputs)  # 文本填充或截取

        data = {"xs": inputs, "ys": labels}
        if filename:
            data_path = os.path.join(self._output_path, filename)
            self._pickle_dump(data, data_path)  # 数据持久化
        return {"x": np.array(data["xs"]), "y": np.array(data["ys"])}


class PredictDataset(DatasetBase):
    def __init__(self, config, device):
        super(PredictDataset, self).__init__(config, device, False)
        self.w2ix, t2ix = self._vocab(None, None)
        self.ix2t = self._inverse_t2ix(t2ix)

        self.stopwords = self._load_stopwords()

    def next_data(self, xs):
        if self.stopwords:
            xs = [[w for w in xs if w not in self.stopwords]]
        else:
            xs = [xs]
        xs = self._trans2ix(xs, self.w2ix)
        xs = self._padding(xs)
        return torch.LongTensor(np.array(xs)).to(self.device)

    def _load_stopwords(self):
        if self._stopwords_path:
            with open(self._stopwords_path, "r", encoding="utf-8") as fr:
                stopwords = [line.strip() for line in fr]
                stopwords = set(stopwords)
                return stopwords
        else:
            return None

    def _inverse_t2ix(self, t2ix):
        ix2t = {v: k for k, v in t2ix.items()}
        return ix2t


if __name__ == '__main__':
    from config import Config

    cfg = Config()
    ds = Dataset(cfg, "cpu")
    xs, ys = ds.gen_data(cfg.train_data)
    for item in ds.next_batch(xs, ys, 4):
        print(item)
        print("-" * 100)
