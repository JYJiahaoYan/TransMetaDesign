import json
import pickle
import warnings
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Union
import os
import numpy as np
import re


class Tokenizer:
    """
    基于 UTF-8 的分词器，包含英文字母、数字（000-999）和标点符号。
    """

    def __init__(self) -> None:
        self.pad_token = "<PAD>"  # 填充标识符
        self.sos_token = "<START>"  # 开始标识符
        self.eos_token = "<END>"  # 结束标识符
        self.unk_token = "<UNK>"  # 未知标识符

        # 特殊标识符的索引
        self.pad_index = 0
        self.sos_index = 1
        self.eos_index = 2
        self.unk_index = 3

        # 偏移量，保证字符的索引从4开始
        self.offset = 4
        char_list = ["double_rec", "cross", "lack_rec", "rec", "ring", "ellipse", "double_ellipse"] + \
                    ["the", "parameter", 'outer_diameter', 'inner_diameter', 'theta', 'phi', 'Px', 'Py', "W1", "L1",
                     "W2", "L2", "major_axis", "minor_axis", "offset", "alpha", "beta", "gamma", "W", "L"] + \
                    list(" .,!?;:'\"()<>[]_=")

        # 负数部分：-999 到 -001，格式为 '-999' 到 '-001'
        negative_numbers = [f"{i:04}" for i in range(-999, 0)]
        # 正数部分：000 到 999，格式为 '000' 到 '999'
        positive_numbers = [f"{i:03}" for i in range(0, 1000)]
        self.char_set = (
                char_list +
                ["PADWORDS" + str(i + len(char_list) + 4) for i in range(1001 - len(char_list) - 4)] +
                negative_numbers +  # 负数部分
                positive_numbers  # 正数部分
        )

        # 创建 token 到索引的映射
        self.token_to_index = {char: idx + self.offset for idx, char in enumerate(self.char_set)}
        self.index_to_token = {idx + self.offset: char for idx, char in enumerate(self.char_set)}
        self.ignore_indices = {self.pad_index, self.sos_index, self.eos_index, self.unk_index}

    def __len__(self):
        # 特��标识符加上字符集的长度
        return len(self.token_to_index) + self.offset

    def char_tokenize(self, text: str) -> List[str]:
        """
        将文本拆分为单个字符。
        """
        return list(text)

    def word_tokenize(self, text: str) -> List[str]:
        """
        使用正则表达式将文本拆分为单词、小数、负数、标点符号和空格。
        """
        # 匹配负小数、负整数、小数、整数、单词、标点符号和空格
        return re.findall(r'-?\d+\.\d+|-?\d+|\w+|[^\w\s]|\s', text, re.UNICODE)

    def encode_int(self, number: str) -> str:
        """
        将整数编码成固定长度的 token，对于负数保留负号，正数不带符号。
        """
        num = int(number)
        if num < 0:
            return f"{num:04}"  # 格式如 '-999'
        else:
            return f"{num:03}"  # 格式如 '000'

    def encode_float(self, number: str) -> str:
        """
        将小数部分编码为 3 位字符串，不转换为整数。
        """
        number = number.ljust(3, '0')[:3]  # 取前三位，不足补零
        return number  # 直接返回字符串

    def encode(self, text: str) -> List[int]:
        """
        把文本转为索引序列，需要在开头加上sos标识符，结尾加上eos标识符，表示这条文本的开始和结束
        """
        indices = [self.sos_index]
        tokens = text.split()
        for token in tokens:
            if self.token_to_index.get(token) is None:

                tokens = self.word_tokenize(token)
                for token in tokens:
                    if re.fullmatch(r'-?\d+', token):
                        # 整数处理，包括负数
                        encoded_token = self.encode_int(token)
                        index = self.token_to_index.get(encoded_token, self.unk_index)
                        indices.append(index)

                    elif re.fullmatch(r'-?\d+\.\d+', token):
                        # 小数处理，包括负数
                        int_part, float_part = token.split(".")
                        encoded_int_part = self.encode_int(int_part)
                        encoded_float_part = self.encode_float(float_part)
                        indices.append(self.token_to_index.get(encoded_int_part, self.unk_index))
                        indices.append(self.token_to_index.get(".", self.unk_index))
                        indices.append(self.token_to_index.get(encoded_float_part, self.unk_index))
                    else:
                        index = self.token_to_index.get(token, self.unk_index)
                        indices.append(index)
            else:
                index = self.token_to_index.get(token, self.unk_index)
                indices.append(index)
        indices.append(self.eos_index)
        return indices

    def decode(self, indices: List[int]) -> str:
        """
        把索引序列重新转换为文本
        """
        chars = []
        i = 0
        while i < len(indices):
            index = indices[i]
            if index == self.eos_index:
                break
            if index < self.offset:
                i += 1
                continue

            char = self.index_to_token.get(index, self.unk_token)
            if re.fullmatch(r'-\d{3}', char) or re.fullmatch(r'\d{3}', char):
                # 整数部分，包括负数和正数
                number = str(int(char))
                # 检查是否有后续的小数点和小数部分
                if i + 2 < len(indices):
                    next_char = self.index_to_token.get(indices[i + 1], self.unk_token)
                    if next_char == ".":
                        float_char = self.index_to_token.get(indices[i + 2], self.unk_token)
                        if re.fullmatch(r'\d{3}', float_char):
                            number += '.' + float_char.rstrip('0')
                            i += 3
                            chars.append(number)
                            continue
                chars.append(number)
                i += 1
            else:
                chars.append(char)
                i += 1
        return ' '.join(chars)


if __name__ == '__main__':
    tokenizer = Tokenizer()
    encoder = tokenizer.encode(
        " ring,the parameter: outer_diameter:100.0,inner_diameter:-31.56,theta:206.0,phi:173.0,Px:311,Py:454")
    print(encoder)
    decoder = tokenizer.decode(encoder)
    print("".join([str(i) for i in decoder]))
    print(len(tokenizer))
