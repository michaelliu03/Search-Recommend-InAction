#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:config_xgb.py
# @Author: Michael.liu
# @Date:2020/7/6 16:48
# @Desc: this code is ....

params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class':2,
    'gamma': 0.1,
    'max_depth': 5,
    'lambda': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'eta': 0.1,
    'seed': 1000,
}