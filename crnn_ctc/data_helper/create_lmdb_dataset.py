# -*- coding: utf-8 -*-

# @Time    : 2019/12/10
# @Author  : Lattine

# ======================
import os
import cv2
import lmdb
import numpy as np
from models import Config
from data_helper import utils


def is_valid(image_bin=None):
    """验证图片是否合规"""
    try:
        image_buf = np.fromstring(image_bin, dtype=np.uint8)
        image = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape[:2]
    except Exception as e:
        print(str(e))
        return False
    else:
        return h * w


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if not isinstance(v, bytes):
                v = v.encode()
            txn.put(k.encode(), v)


def create_dataset(lmdb_output, images, texts, converter=None, lmdb_size=1024 ** 3, flush_every=1000):
    # lmdb_size默认1G
    assert len(images) == len(texts), "length of images and texts should be equal."
    n_samples = len(images)
    env = lmdb.open(lmdb_output, map_size=lmdb_size)
    cache = {}
    cnt = 0  # 对图片建立索引
    for ix in range(n_samples):
        image_path = images[ix]
        text = texts[ix]

        with open(image_path, "rb") as fr:
            image_bin = fr.read()  # 读取图片的二进制流
            if not is_valid(image_bin): continue  # 验证图片是否合规

        # text和label的关系是，label是构成text的字符序列，length指的是label的长度
        label, length = converter.encode(text)
        label, length = str(label.tolist()), str(length.tolist())

        # 以下问LMDB中的KEYs
        image_key = "image-%09d" % cnt
        text_key = 'text-%09d' % cnt
        label_key = "label-%09d" % cnt
        length_key = 'length-%09d' % cnt
        cache[image_key] = image_bin
        cache[text_key] = text
        cache[label_key] = label
        cache[length_key] = length

        if cnt % flush_every == 0 and cnt > 0:
            write_cache(env, cache)
            cache = {}
            print(f"Written {cnt} / {n_samples}")

        cnt += 1
    n_samples = cnt  # 实际存储的数据量（排除异常图片）
    cache["total-samples"] = str(n_samples)
    write_cache(env, cache)
    print(f"Created dataset with {n_samples} samples")


def load_data(image_label_file, image_prefix):
    """打开图片、标签对文件，读取文件目录、标签"""
    with open(image_label_file, encoding="utf-8") as fr:
        image_list, text_list = [], []
        for line in fr:
            segs = line.strip().split("\t")
            image_list.append(os.path.join(image_prefix, segs[0].strip()))
            text_list.append(segs[1].strip())
        return image_list, text_list


if __name__ == '__main__':
    cfg = Config()
    converter = utils.StrLabelConverter(cfg.alphabets, ignore_case=True)
    # ------------------- train dataset lmdb -------------------
    train_lmdb_output = cfg.train_lmdb
    if not os.path.exists(train_lmdb_output): os.makedirs(train_lmdb_output)
    image_list, text_list = load_data(cfg.train_image_label_file, cfg.train_image_prefix)
    create_dataset(train_lmdb_output, image_list, text_list, converter=converter, lmdb_size=1023**3*3)

    # ------------------- eval dataset lmdb -------------------
    eval_lmdb_output = cfg.eval_lmdb
    if not os.path.exists(eval_lmdb_output): os.makedirs(eval_lmdb_output)
    image_list, text_list = load_data(cfg.eval_image_label_file, cfg.eval_image_prefix)
    create_dataset(eval_lmdb_output, image_list, text_list, converter=converter)
