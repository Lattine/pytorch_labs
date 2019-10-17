# -*- coding: utf-8 -*-

# @Time    : 2019/10/16
# @Author  : Lattine

# ======================
import os

import cv2
import lmdb
import numpy as np

from config import Config


class LmdbDataset:
    def __init__(self, config):
        self.config = config
        self.map_size = self.config.map_size
        self.save_every = 1000  # 每1000个保存一次

    def create_lmdb(self):
        self._generate_img_label_file()
        train_image_list, train_label_list = self._load_data(self.config.train_data_path, self.config.train_data_img_label_file)
        self._create_lmdb_dataset(train_image_list, train_label_list, self.config.train_data_lmdb)
        test_image_list, test_label_list = self._load_data(self.config.test_data_path, self.config.test_data_img_label_file)
        self._create_lmdb_dataset(test_image_list, test_label_list, self.config.test_data_lmdb)

    def _generate_img_label_file(self):
        # 构件训练集图片与标签文件字典
        with open(self.config.train_data_img_label_file, "w+", encoding="utf-8") as fr:
            for label in os.listdir(self.config.train_data_path):
                for fn in os.listdir(os.path.join(self.config.train_data_path, label)):
                    if fn.endswith(".jpg") or fn.endswith(".png"):
                        fr.write(f"{fn}\t{label}\n")

        # 构建测试集图片与标签文件字典
        with open(self.config.test_data_img_label_file, "w+", encoding="utf-8") as fr:
            for label in os.listdir(self.config.test_data_path):
                for fn in os.listdir(os.path.join(self.config.test_data_path, label)):
                    if fn.endswith(".jpg") or fn.endswith(".png"):
                        fr.write(f"{fn}\t{label}\n")

    def _load_data(self, path, img_label_file):
        image_list = []
        label_list = []
        with open(img_label_file, encoding="utf-8") as fr:
            for line in fr:
                try:
                    name, label = line.strip().split("\t")
                    img_path = path + label + "/" + name if path[-1] == "/" else path + "/" + label + "/" + name
                    image_list.append(img_path)
                    label_list.append(label)
                except:
                    print(f"Error with line: {line}")
        return image_list, label_list

    def _create_lmdb_dataset(self, image_list, label_list, lmdb_path):
        assert len(image_list) == len(label_list), "length of images must equal with labels."
        if not os.path.exists(lmdb_path): os.makedirs(lmdb_path)  # 确保数据库目录存在

        n_samples = len(image_list)
        env = lmdb.open(lmdb_path, map_size=self.map_size)
        cache = {}
        cnt = 1
        for i in range(n_samples):
            image_key = "image-%09d" % cnt
            label_key = "label-%09d" % cnt

            with open(image_list[i], "rb") as fr:
                image_bin = fr.read()

            if not self._check_valid(image_bin):
                continue

            cache[image_key] = image_bin
            cache[label_key] = label_list[i]

            if cnt % self.save_every == 0:
                self._write_cache(env, cache)
                cache = {}
                print("Written %d / %d" % (cnt, n_samples))
            cnt += 1
        n_samples = cnt - 1
        cache["n_samples"] = str(n_samples)
        self._write_cache(env, cache)
        print("Created dataset with %d samples" % n_samples)

    def _check_valid(self, image_bin):
        if image_bin is None: return False
        try:
            image_buf = np.fromstring(image_bin, dtype=np.uint8)
            img = cv2.imdecode(image_buf, cv2.IMREAD_ANYCOLOR)
            h, w = img.shape[:2]
        except:
            return False
        else:
            if h * w == 0:
                return False
        return True

    def _write_cache(self, env, cache):
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                if not isinstance(v, bytes):
                    v = v.encode()
                txn.put(k.encode(), v)


if __name__ == '__main__':
    config = Config()
    db = LmdbDataset(config)
    db.create_lmdb()
