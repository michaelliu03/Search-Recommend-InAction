#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:helper.py
# @Author: Michael.liu
# @Date:2020/6/15 18:59
# @Desc: this code is ....
import json
from .data_preprocess import *

class DssmData(object):

    @staticmethod
    def load_vocab(file_path):
        word_dict = {}
        with open(file_path, encoding='utf8') as f:
            for idx, word in enumerate(f.readlines()):
                word = word.strip()
                word_dict[word] = idx
        return word_dict

    @staticmethod
    def load_data(file_path):
        """
        gen datasets, convert word into word ids.
        :param file_path:
        :return: [[query, pos sample, 4 neg sample]], shape = [n, 6]
        """
        data_map = {'query': [], 'query_len': [], 'doc_pos': [], 'doc_pos_len': [], 'doc_neg': [], 'doc_neg_len': []}
        with open(file_path, encoding='utf8') as f:
            for line in f.readlines():
                spline = line.strip().split('\t')
                if len(spline) < 4:
                    continue
                prefix, query_pred, title, tag, label = spline
                if label == '0':
                    continue
                cur_arr, cur_len = [], []
                query_pred = json.loads(query_pred)
                # only 4 negative sample
                for each in query_pred:
                    if each == title:
                        continue
                    cur_arr.append(convert_word2id(each, conf.vocab_map))
                    each_len = len(each) if len(each) < conf.max_seq_len else conf.max_seq_len
                    cur_len.append(each_len)
                if len(cur_arr) >= 4:
                    data_map['query'].append(convert_word2id(prefix, conf.vocab_map))
                    data_map['query_len'].append(len(prefix) if len(prefix) < conf.max_seq_len else conf.max_seq_len)
                    data_map['doc_pos'].append(convert_word2id(title, conf.vocab_map))
                    data_map['doc_pos_len'].append(len(title) if len(title) < conf.max_seq_len else conf.max_seq_len)
                    data_map['doc_neg'].extend(cur_arr[:4])
                    data_map['doc_neg_len'].extend(cur_len[:4])
                pass
        return data_map


