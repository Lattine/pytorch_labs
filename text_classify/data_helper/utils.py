# -*- coding: utf-8 -*-

# @Time    : 2019/11/1
# @Author  : Lattine

# ======================
import os


def remove_files(path):
    for fn in os.listdir(path):
        fpath = os.path.join(path, fn)  # 构造文件路径
        if os.path.isfile(fpath):  # 文件
            os.remove(fpath)
        else:  # 文件夹
            remove_files(fpath)  # 递归的删除子文件夹
