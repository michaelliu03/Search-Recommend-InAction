#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:lr_model.py
# @Author: Michael.liu
# @Date:2020/6/17 14:51
# @Desc: this code is ....

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib


class LrModel(object):
    train_df = None
    test_df = None
    vali_df = None
    model = None

    def __init__(self, trainfile, testfile, valifile):
        self.train_file = trainfile
        self.test_file = testfile
        self.vali_file = valifile

    def load_train_data(self, names):
        self.train_df = pd.read_csv(self.train_file, header=0, sep=',')

    def load_vali_data(self, names):
        self.vali_df = pd.read_csv(self.vali_file,header=0, sep=',')

    def load_test_data(self, names):
        self.test_df = pd.read_csv(self.test_file, header=0, sep=',')

    def train(self, feature_head, target_head):
        '''
        :param train_df: dataframe of train data
        :param vali_df:  dataframe of valid data
        :param test_df:  dataframe of test data
        :param feature_head:  list of features names for model
        :param target_head:  str of target name for model
        :return:
        '''

        #print(self.train_df)
        #print(feature_head)
        x_train = self.train_df[feature_head]
        y_train = self.train_df[target_head]

        # 暂时没有用到验证集
        # todo：可以random 训练和验证数据集
        # x_vali = vali_df[feature_head]
        # y_vali = vali_df[target_head]

        x_test = self.test_df[feature_head]
        y_test = self.test_df[target_head]

        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        print('Accuracy of LR Classifier:%f' % lr.score(x_test, y_test))
        joblib.dump(lr, 'gen_lr.pkl')
        self.model = lr

    # 加载模型并进行预测
    def infer(self, feature_head, target_head, model_path=None):
        '''
        :param feature_head: list names of features for model
        :param target_head: string name of target for model
        :param model_path : model path for loading model (model must have same feature head and target head with valid data)
        :return:
        '''
        if model_path != None:
            self.model = joblib.load(model_path)

        x_vali = self.vali_df[feature_head]
        y_vali = self.vali_df[target_head]
        print('Accuracy of LR Classifier:%f' % self.model.score(x_vali, y_vali))