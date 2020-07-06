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
from .config_xgb import *

from .xgboost_model import *

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
    ################################ DT   ############################
    # decisiontTree_model = DecisionTreeModel(args)
    # decisiontTree_model.load_train_data(head)
    # decisiontTree_model.load_test_data(head)
    # decisiontTree_model.train(head[2:],head[0])

    ############################### gbdt ############################
    # gbdt_model = GBDT_LR_MODEl(args)
    # gbdt_model.load_train_data(head)
    # gbdt_model.load_test_data(head)
    # gbdt_model.train(head[0:],head[0])

    ############################## xgboost###########################
    params = params

    train_data_path = 'format_train.txt'
    test_data_path = 'format_test.txt'
    target = 'relevent'
    ignore_list = ['qid']
    sep = ','
    xg = xgb_model(params=params)

    x_train, y_train, x_test, y_test = xg.load_data(train_data_path, test_data_path, target, sep, ignore_list)

    print(x_train.info())

    xg.fit(x_train, y_train, x_test, y_test, rounds=500)

print("......训练结束.....,共耗费 %s " % (time.time()-t_start))

