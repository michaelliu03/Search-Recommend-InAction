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

head = 'index,qid,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35,f36,f37,f38,f39,f40,f41,f42,f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65,f66,f67,f68,f69,f70,f71,f72,f73,f74,f75,f76,f77,f78,f79,f80,f81,f82,f83,f84,f85,f86,f87,f88,f89,f90,f91,f92,f93,f94,f95,f96,f97,f98,f99,f100,f101,f102,f103,f104,f105,f106,f107,f108,f109,f110,f111,f112,f113,f114,f115,f116,f117,f118,f119,f120,f121,f122,f123,f124,f125,f126,f127,f128,f129,f130,f131,f132,f133,f134,f135,f136'
head = head.split(',')


class Lr_model(object):

    def __init__(self,trainfile,testfile,valifile):
        self.train_file = trainfile
        self.test_file = testfile
        self.vali_file = valifile

    def load_train_data(self,train_file):
        train_df = pd.read_csv(train_file, header=0, sep=',')
        return train_df

    def load_vali_data(self,vali_file):
        vali_df = pd.read_csv(vali_file,header=0,sep=',')
        return vali_df

    def load_test_data(self,test_file):
        test_df =pd.read_csv(test_file,header=0,sep=',')
        return test_df

    def train(self,train_df,vali_df,test_df):
        x_train = train_df[head[2:]]
        y_train = train_df[head[1]]

        #暂时没有用到验证集
        #todo：可以random 训练和验证数据集
        x_vali = vali_df[head[3:]]
        y_vali = vali_df[head[2]]

        x_test = test_df[head[3:]]
        y_test = test_df[head[2]]

        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        print('Accuracy of LR Classifier:%f' % lr.score(x_test, y_test))
        joblib.dump(lr, 'gen_lr.pkl')

    # 加载模型并进行预测
    def infer(self,vali_df):
        print("this is infor")

