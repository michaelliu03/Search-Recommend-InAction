#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:decisiontree_model.py
# @Author: Michael.liu
# @Date:2020/6/30 12:28
# @Desc: this code is ....

import pandas as pd
import time
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

class DecisionTreeModel(object):
    train_df = None
    vali_df = None
    test_df = None
    model = None

    def __init__(self,args):
        self.args = args
        with open(args.config_path, "r", encoding="utf8") as fr:
            self.config = json.load(fr)

        self.train_file = self.config["dt_train_file"]
        self.vali_file= self.config["dt_vali_file"]
        self.test_file = self.config["dt_test_file"]
        self.criterion= self.config["criterion"]

    def load_train_data(self, names):
        self.train_df = pd.read_csv(self.train_file, names=names, sep=',')

    def load_vali_data(self, names):
        self.vali_df = pd.read_csv(self.vali_file, names=names, sep=',')

    def load_test_data(self, names):
        self.test_df = pd.read_csv(self.test_file, names=names, sep=',')



    def train(self, feature_head, target_head):
        t_start = time.time()
        X_train = self.train_df[feature_head]
        y_train = self.train_df[target_head]

        X_test = self.test_df[feature_head]
        y_test = self.test_df[target_head]

        clf = DecisionTreeClassifier(criterion=self.criterion)
        clf.fit(X_train, y_train)
        print('Accuracy of Adaboost Classifier:%f' % bdt.score(X_test, y_test))
        joblib.dump(clf, 'gen_dt.pkl')
        self.model = clf
        print('cost the time %s' % (time.time() - t_start))



