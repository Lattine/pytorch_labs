# -*- coding: utf-8 -*-

# @Time    : 2019/9/18
# @Author  : Lattine
# ======================
import os
import uuid
import random
from PIL import Image, ImageDraw, ImageFont
from config import Config


class GeneratorForCRNN:
    def __init__(self, img_h, img_w, fonts_path):
        self.type = "date"
        self.WIDTH = img_w
        self.HEIGHT = img_h
        self.fonts_path = fonts_path
        self.fonts = self._load_fonts()  # 加载字体文件
        self.bgcolor = (255, 255, 255)  # 背景颜色，默认为白色
        self.fontcolor = (0, 0, 0)  # 字体颜色，默认为黑色

    def generate(self, output_path, output_file):
        self.type = random.choice(types)
        if self.type == "date":
            val = self._gen_date()
        elif self.type == "words":
            val = self._gen_words()
        elif self.type == "nums":
            val = self._gen_nums()
        image = self._gene_code(random.choice(self.fonts), val)
        name = f"{str(uuid.uuid4())}.jpg"
        name = name.replace("-", "")
        image.save(os.path.join(output_path, name))
        self._log(name, val, output_file)

    def _log(self, name, val, output_file):
        with open(output_file, "a+", encoding="utf-8") as fw:
            fw.write(f"{name}\t{val}\n")

    def _gen_date(self):
        day = random.randint(1, 32)
        month = random.randint(1, 13)
        year = random.randint(2000, 2100)
        val = f"{day}/{month}/{year}"
        return val

    def _gen_words(self):
        n = random.randint(2, 5)
        val = "".join(random.sample(ALPHA, n))
        return val

    def _gen_nums(self):
        m = random.randint(1, 20)
        n = "".join(random.sample(nums, 4))
        return f"{m}.{n}%"

    def _load_fonts(self):
        fonts = []
        for fn in os.listdir(self.fonts_path):
            if fn.endswith(".ttf"):
                fonts.append(os.path.join(self.fonts_path, fn))
        return fonts

    # 生成验证码
    def _gene_code(self, aFont, text):
        number = len(text)
        width, height = (self.WIDTH, self.HEIGHT)  # 生成验证码图片的宽度和高度
        image = Image.new('RGB', (width, height), self.bgcolor)  # 创建图片
        font = ImageFont.truetype(aFont, self.HEIGHT - random.randint(2, 7))  # 验证码的字体
        draw = ImageDraw.Draw(image)  # 创建画笔
        font_width, font_height = font.getsize(text)
        draw.text(((width - font_width) / number, (height - font_height) / number), text, font=font, fill=self.fontcolor)  # 填充字符串
        # image = image.transform((width + 30, height + 10), Image.AFFINE, (1, -0.3, 0, -0.1, 1, 0), Image.BILINEAR)  # 创建扭曲
        # image = image.transform((width+20,height+10), Image.AFFINE, (1,-0.3,0,-0.1,1,0),Image.BILINEAR)  #创建扭曲
        # image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)  # 滤镜，边界加强
        return image


def remove_files(path):
    for fn in os.listdir(path):
        fpath = os.path.join(path, fn)  # 构造文件路径
        if os.path.isfile(fpath):  # 文件
            os.remove(fpath)
        else:  # 文件夹
            remove_files(fpath)  # 递归的删除子文件夹


if __name__ == '__main__':
    types = ["date", "words", "nums"]
    nums = "0123456789"
    ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    cfg = Config()
    g = GeneratorForCRNN(img_h=cfg.img_h, img_w=cfg.img_w, fonts_path=cfg.fonts_path)

    train_output_path = cfg.train_input_file_prefix
    remove_files(train_output_path)
    train_output_file = cfg.train_input_file_with_image_label
    for i in range(cfg.train_examples):
        g.generate(train_output_path, train_output_file)

    test_output_path = cfg.test_input_file_prefix
    remove_files(test_output_path)
    test_output_file = cfg.test_input_file_with_image_label
    for i in range(cfg.test_examples):
        g.generate(test_output_path, test_output_file)
