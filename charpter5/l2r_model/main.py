#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:main.py
# @Author: Michael.liu
# @Date:2020/6/18 15:30
# @Desc: this code is ....
from .lr_model import *
from .helper import *
import time


head_path = './feature_head.txt'
head = load_head(head_path)

print(head)

if __name__ == '__main__':
    #global head
    print("......开始训练.....")
    lr_model = LrModel('./format_train.txt','./format_vali.txt','./format_vali.txt')
    lr_model.load_train_data(head)
    lr_model.load_test_data(head)
    lr_model.train(head[2:],head[0])
    print("......训练结束.....")