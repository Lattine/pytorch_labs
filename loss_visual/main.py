# -*- coding: UTF-8 -*-

import os
from visual import Visualizer


class Losser:
    def __init__(self):
        self.base_path = os.path.abspath(os.path.dirname(__file__))
        self.files = os.listdir(os.path.join(self.base_path, 'losses'))

        self._build()

    def _build(self):
        for filename in self.files:
            vis = Visualizer(filename[:-4])
            self._paint(vis, os.path.join(self.base_path, "losses/" + filename))
            print("{} is done.".format(filename[:-4]))

    def _paint(self, vis, filepath):
        with open(filepath, encoding="utf-8") as fin:
            for line in fin:
                segs = line.split("\t")
                if len(segs) != 3: continue
                vis.plot('loss', float(segs[-1]))


if __name__ == '__main__':
    l = Losser()
