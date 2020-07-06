#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:gbdt_lr_model.py
# @Author: Michael.liu
# @Date:2020/6/30 14:10
# @Desc: this code is ....

import json
import time
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 63,
	'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

class GBDT_LR_MODEl(object):
    train_df = None
    vali_df = None
    test_df = None
    model = None

    def __init__(self, args):
        self.args = args
        with open(args.config_path, "r", encoding="utf8") as fr:
            self.config = json.load(fr)

        self.train_file = self.config["dt_train_file"]
        self.vali_file = self.config["dt_vali_file"]
        self.test_file = self.config["dt_test_file"]

    def load_train_data(self, names):
        self.train_df = pd.read_csv(self.train_file, header=0, sep=',')

    def load_vali_data(self, names):
        self.vali_df = pd.read_csv(self.vali_file, header=0, sep=',')

    def load_test_data(self, names):
        self.test_df = pd.read_csv(self.test_file, header=0, sep=',')

    def train(self, feature_head, target_head):
        t_start = time.time()
        X_train = self.train_df[feature_head]
        y_train = self.train_df[target_head]

        X_test = self.test_df[feature_head]
        y_test = self.test_df[target_head]



        lgb_train = lgb.Dataset(X_train, y_train)
        #lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=100,
                        valid_sets=lgb_train)

        #print('Accuracy of gbdt_lr Classifier:%f' % gbm.score(X_test, y_test))

        gbm.save_model('model.txt')
