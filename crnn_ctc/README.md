### 项目描述

- 基于CRNN+BiLSTM+CTC的面向中英文的OCR识别

---
### 运行  
- 修改`models`下对应模型的Config配置，设置好数据集目录；  
- 运行 `data_helper/build_image_label_file.py`，将根据数据集目录生成图片标签对文件，用于构件LMDB数据库；  
- 运行 `data_helper/create_lmdb_dataset.py`，将根据标签对文件生成训练集、验证集对应的LMDB数据库；  