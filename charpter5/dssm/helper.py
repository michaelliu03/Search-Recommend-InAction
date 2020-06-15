#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:helper.py
# @Author: Michael.liu
# @Date:2020/6/15 18:59
# @Desc: this code is ....


def load_vocab(file_path):
    word_dict = {}
    with open(file_path, encoding='utf8') as f:
        for idx, word in enumerate(f.readlines()):
            word = word.strip()
            word_dict[word] = idx
    return word_dict