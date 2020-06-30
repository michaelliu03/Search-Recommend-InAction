#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:main.py
# @Author: Michael.liu
# @Date:2020/6/30 13:18
# @Desc: this code is ....


from .helper import *
from .decisiontree_model import *
from .gbdt_lr_model import *
import time
import argparse

head_path = './feature_head.txt'
head = load_head(head_path)
print(head)


if __name__ == '__main__':
    #global head
    t_start = time.time()
    print("......开始训练.....")
    print("....start....")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="config path of model")
    args = parser.parse_args()
    # decisiontTree_model = DecisionTreeModel(args)
    # decisiontTree_model.load_train_data(head)
    # decisiontTree_model.load_test_data(head)
    # decisiontTree_model.train(head[2:],head[0])
    gbdt_model = GBDT_LR_MODEl(args)
    gbdt_model.load_train_data(head)
    gbdt_model.load_test_data(head)
    gbdt_model.train(head[0:],head[0])
    print("......训练结束.....,共耗费 %s " % (time.time()-t_start))

