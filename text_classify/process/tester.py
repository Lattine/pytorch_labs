# -*- coding: utf-8 -*-

# @Time    : 2019/10/29
# @Author  : Lattine

# ======================
import os
import pandas as pd
import torch

from config import Config
from data_helper import PredictDataset
from models import Model


class Tester:
    def __init__(self, config):
        self.config = config
        self.device = "cpu"

        self._load_data()
        self.config.vocab_size = self.vocab_size
        self.nets = Model(self.config, None)
        print(self.nets)
        self._load_model(self.config.saved_model)

    def predict(self):
        result = []
        for data in self.test_data:
            # x = self.dataset.next_data(data["comment_cutted"])
            x = self.dataset.next_data(data["content"])
            pred = self.nets(x)
            probs = torch.nn.functional.softmax(pred, -1)
            val, ix = probs.max(-1)
            item = {"id": data["id"], "label": self.ix2t.get(ix.item())}
            result.append(item)
        return result

    def to_csv(self, datas):
        res = {"id": [], "label": []}
        for data in datas:
            res["id"].append(data["id"])
            res["label"].append(data["label"])

        df = pd.DataFrame(res)
        df.to_csv(self.config.predict_result, index=False)
        print(f"Result is saved in : {self.config.predict_result}")

    def _load_data(self):
        self.test_data = self._read_raw_data(self.config.test_data)
        self.dataset = PredictDataset(self.config, self.device)
        self.vocab_size = self.dataset.vocab_size
        self.ix2t = self.dataset.ix2t

    def _read_raw_data(self, path):
        data = pd.read_csv(path)
        data["content"] = data["comment"].apply(lambda text: list(text))
        test_data = []
        for index, row in data.iterrows():
            item = {}
            item["id"] = row["id"]
            item["content"] = row["content"]
            test_data.append(item)
        return test_data

    def _load_model(self, path):
        if os.path.exists(path):
            self.nets.load_state_dict(torch.load(path))
            print("Load pretrained model.")
        else:
            print("Model don't exist!")


if __name__ == '__main__':
    config = Config()
    p = Tester(config)
    data = p.predict()
    p.to_csv(data)
