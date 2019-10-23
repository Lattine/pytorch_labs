# -*- coding: utf-8 -*-

# @Time    : 2019/10/10
# @Author  : Lattine

# ======================
import os
import cv2
import lmdb
import numpy as np
from config import Config
from data_helper import utils


def check_iamge_valid(image_bin):
    if image_bin is None:
        return False
    try:
        image_buf = np.fromstring(image_bin, dtype=np.uint8)
        img = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]
    except:
        return False
    else:
        if h * w == 0:
            return False
    return True


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if not isinstance(v, bytes):
                v = v.encode()
            txn.put(k.encode(), v)


def create_dataset(output, images, texts, lexicon_list=None, check_valid=True, cvt=None):
    assert (len(images) == len(texts))
    n_samples = len(images)
    env = lmdb.open(output, map_size=9511627776)
    cache = {}
    cnt = 1
    for i in range(n_samples):
        image_path = images[i].split()[0].replace("\n", "").replace("\r\n", "")
        text = "".join(texts[i])
        with open(image_path, "rb") as fr:
            image_bin = fr.read()
        if check_valid:
            pass

        image_key = "image-%09d" % cnt
        label_key = "label-%09d" % cnt
        length_key = 'length-%09d' % cnt
        text_key = 'text-%09d' % cnt

        label, length = converter.encode(text)
        label, length = str(label.tolist()), str(length.tolist())

        cache[image_key] = image_bin
        cache[label_key] = label
        cache[length_key] = length
        cache[text_key] = text

        if lexicon_list:
            lexicon_key = "lexicon-%09d" % cnt
            cache[lexicon_key] = " ".join(lexicon_list[i])
        if cnt % 1000 == 0:
            write_cache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, n_samples))
        cnt += 1
    n_samples = cnt - 1
    cache["num-samples"] = str(n_samples)
    write_cache(env, cache)
    print('Created dataset with %d samples' % n_samples)


if __name__ == '__main__':
    cfg = Config()
    # ------------------- train lmdb -------------------
    output_path = cfg.train_output_data_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(cfg.train_input_file_with_image_label, encoding="utf-8") as fr:
        image_list = []
        text_list = []
        for line in fr:
            pts = line.strip().split("\t")
            image_list.append(os.path.join(cfg.train_input_file_prefix, pts[0]))
            text_list.append(pts[1])
        converter = utils.StrLabelConverter(cfg.alphabets, ignore_case=True)
        create_dataset(output_path, image_list, text_list, cvt=converter)

    # ------------------- test lmdb ---------------------
    output_path = cfg.test_output_data_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(cfg.test_input_file_with_image_label, encoding="utf-8") as fr:
        image_list = []
        text_list = []
        for line in fr:
            pts = line.strip().split("\t")
            image_list.append(os.path.join(cfg.test_input_file_prefix, pts[0]))
            text_list.append(pts[1])
        converter = convert.StrLabelConverter(cfg.alphabets, ignore_case=True)
        create_dataset(output_path, image_list, text_list, cvt=converter)