#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:query_intent_fasttext.py
# @Author: Michael.liu
# @Date:2020/5/14 19:38
# @Desc: this code is ....

import logging
import os
import numpy as np


from random import shuffle
import fasttext.FastText as fasttext

train_file_path = u"../../data/chapter4/intent/query_intent_train_used.csv"
test_file_path = u"../../data/chapter4/intent/query_intent_test_used.csv"


def train_model(ipt=None, opt=None, model='', dim=100, epoch=5, lr=0.1, loss='softmax'):
    np.set_printoptions(suppress=True)
    if os.path.isfile(model):
        classifier = fasttext.load_model(model)
    else:
        classifier = fasttext.train_supervised(ipt, label='__label__', dim=dim, epoch=epoch,
                                         lr=lr, wordNgrams=2, loss=loss)

        classifier.save_model(opt)
    return classifier

dim = 100
lr = 5
epoch = 20
model = f'data_dim{str(dim)}_lr0{str(lr)}_iter{str(epoch)}.model'

classifier = train_model(ipt=train_file_path,
                         opt=model,
                         model=model,
                         dim=dim, epoch=epoch, lr=0.5
                         )

result = classifier.test(test_file_path)
# print(result)
#
#
# texts = []
# labels_right = []
# labels_predict = []
# i = 0
# top_num = 10

# with open(test_file_path,'r',encoding='utf-8') as fr:
#     lines = fr.readlines()
#
# for line in lines:
#     labels_right.append(line.split("__label__")[1].rstrip().replace("\n",""))
#     texts.append(line.split("__label__")[0].rstrip().replace("\t",""))
#
#
# labels_predict = classifier.predict(texts, top_num)
# correct_num = 0
# wrong_num = 0
#
# for i in range(0,len(labels_right)):
#    if labels_right[i] in labels_predict[i]:
#       correct_num += 1
#    else:
#       wrong_num +=1
#
# accuracy = correct_num/(correct_num+wrong_num)
#
# print("top 10 precision")
# print(accuracy)
