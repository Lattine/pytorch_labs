# -*- coding: utf-8 -*-

# @Time    : 2019/10/29
# @Author  : Lattine

# ======================
import os
import pandas as pd
import torch
import jieba
import re
from data_helper import PredictDataset
from models import text_cnn as nets


class Tester:
    def __init__(self):
        self.config = nets.Config()
        self.config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init()  # 初始化数据集，动态调整参数

        self.test_data = []
        self._read_data(self.config.test_data)
        self.config.vocab_size = self.vocab_size
        self.model = nets.Model(self.config, None)
        print(self.model)
        self._load_model(self.config.saved_model)
        self.model.to(self.config.device)

    def predict(self):
        for data in self.test_data:
            x = self.dataset.next_data(data["content"])
            pred = self.model(x)
            probs = torch.nn.functional.softmax(pred, -1)
            val, ix = probs.max(-1)
            data["proba"] = val.item()
            data["prediction"] = self.ix2t.get(ix.item())

        self.predict_regex()

    def _init(self):
        self.dataset = PredictDataset(self.config)
        self.vocab_size = self.dataset.vocab_size
        self.ix2t = self.dataset.ix2t

    def _load_model(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            print("Load pretrained model.")
        else:
            print("Model don't exist!")

    # ---------- 需要依据数据集重构 ----------
    def _read_data(self, path):
        raw = pd.read_csv(path)
        print(raw.head())
        # raw["content"] = raw["content"].apply(lambda x: jieba.lcut(re.sub(r"【.*?】|（.*?）|\d+|%|\.", "", x)))
        raw["content"] = raw["comment"].apply(lambda s: [s[i:i + n] for n in range(1, 3 + 1) for i in range(len(s) - n + 1)])
        for ix, row in raw.iterrows():
            item = {
                "content": row["content"],
                "id": row["id"],
            }
            self.test_data.append(item)

    def predict_regex(self):
        pass

    def post_result(self):
        # raw = {"prediction": [], "target": [], "proba": [], "content": []}
        raw = {"id": [], "label": []}
        for data in self.test_data:
            raw["id"].append(data["id"])
            raw["label"].append(data["prediction"])
            # raw["proba"].append(data["proba"])
            # raw["content"].append(data["content"])

        df = pd.DataFrame(raw)
        df.to_csv(self.config.predict_result, index=False)
        print(f"Result is saved in : {self.config.predict_result}")


if __name__ == '__main__':
    p = Tester()
    p.predict()
    p.post_result()
