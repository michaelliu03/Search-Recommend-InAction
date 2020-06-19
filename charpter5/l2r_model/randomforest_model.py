#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:randomforest_model.py
# @Author: Michael.liu
# @Date:2020/6/18 17:43
# @Desc: this code is random forest model

import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import time
from sklearn.externals import joblib

class random_forest_model():
    train_df = None
    vali_df = None
    test_df = None
    model = None
    config = None

    def __init__(self, args):
        self.args = args
        with open(args.config_path, "r", encoding="utf8") as fr:
            self.config = json.load(fr)

        self.train_file = self.config["rf_train_file"]
        self.vali_file = self.config["rf_vali_file"]
        self.test_file = self.config["rf_test_file"]

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


        rf =  RandomForestClassifier(n_estimators=self.config['n_estimators'],max_depth=self.config['max_depth'],min_samples_split=self.config['min_samples_split'],
                                     min_samples_leaf=self.config['min_samples_leaf'],max_leaf_nodes=self.config['max_leaf_nodes'],bootstrap=self.config['bootstrap'],
                                     n_jobs=self.config['n_jobs'],min_weight_fraction_leaf=self.config['min_weight_fraction_leaf'],criterion=self.config['criterion'],random_state=self.config['random_state'])


        rf.fit(x_train, y_train)
        print('Accuracy of Adaboost Classifier:%f' % rf.score(x_test, y_test))
        joblib.dump(rf, 'gen_bdt.pkl')
        self.model = rf
        print('cost the time %s' % (time.time() - t_start))

    def infer(self, feature_head, target_head, model_path=None):
        t_start = time.time()
        if model_path != None:
            self.model = joblib.load(model_path)

        x_vali = self.vali_df[feature_head]
        y_vali = self.vali_df[target_head]
        print('Accuracy of Adaboost Classifier:%f' % self.model.score(x_vali, y_vali))

