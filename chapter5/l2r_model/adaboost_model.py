#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:adaboost_model.py
# @Author: Michael.liu
# @Date:2020/6/18 17:36
# @Desc: this code is ....


import numpy as np
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.externals import joblib

class AdaBoostModel(object):
    train_df = None
    vali_df = None
    test_df = None
    model = None

    def __init__(self,args):
        self.args = args
        with open(args.config_path, "r", encoding="utf8") as fr:
            self.config = json.load(fr)

        self.train_file = self.config["adt_train_file"]
        self.vali_file= self.config["adt_vali_file"]
        self.test_file = self.config["adt_test_file"]

    def load_train_data(self, names):
        self.train_df = pd.read_csv(self.train_file, names=names, sep=',')

    def load_vali_data(self, names):
        self.vali_df = pd.read_csv(self.vali_file, names=names, sep=',')

    def load_test_data(self, names):
        self.test_df = pd.read_csv(self.test_file, names=names, sep=',')

    def train(self, feature_head, target_head):
        t_start = time.time()
        x_train = self.train_df[feature_head]
        y_train = self.train_df[target_head]

        x_test = self.test_df[feature_head]
        y_test = self.test_df[target_head]

        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=self.config["adt_max_depth"],
                                                        min_samples_split=self.config["adt_min_samples_split"],
                                                        min_samples_leaf=self.config["adt_min_samples_leaf"]),
                                 algorithm = self.config["adt_algorithm"],
                                 n_estimators = self.cofig["adt_n_estimators"],
                                 learning_rate = self.config["adt_learning_rate"])
        bdt.fit(x_train, y_train)
        print('Accuracy of Adaboost Classifier:%f' % bdt.score(x_test, y_test))
        joblib.dump(bdt, 'gen_bdt.pkl')
        self.model = bdt
        print('cost the time %s' % (time.time() - t_start))

    def infer(self, feature_head, target_head, model_path=None):
        t_start = time.time()
        if model_path != None:
            self.model = joblib.load(model_path)

        x_vali = self.vali_df[feature_head]
        y_vali = self.vali_df[target_head]
        print('Accuracy of Adaboost Classifier:%f' % self.model.score(x_vali, y_vali))









