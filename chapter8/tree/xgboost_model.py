#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:xgboost_model.py
# @Author: Michael.liu
# @Date:2020/7/1 13:19
# @Desc: this code is ....
import  xgboost as xgb
from .config_xgb import  *
import pandas as pd
from sklearn.metrics import  accuracy_score


class xgb_model():
    params = {}

    def __init__(self,params):
        self.params= dict(params.items())

    def load_data(self,train_data_path,test_data_path,target,sep,ignore_list=None):
        train_data = pd.read_csv(train_data_path,header=0,sep=sep)
        test_data = pd.read_csv(test_data_path,header=0,sep=sep)
        y_train = train_data[target]
        y_test = test_data[target]

        drop_list = [target]

        if ignore_list != None :
            drop_list = drop_list + ignore_list

        x_train = train_data.drop(columns=drop_list)
        x_test = test_data.drop(columns=drop_list)

        return x_train,y_train,x_test,y_test



    def fit(self,x_train,y_train,x_test,y_test,rounds):

        data_train = xgb.DMatrix(x_train,y_train)

        print(self.params)

        model = xgb.train(params=self.params,dtrain=data_train,num_boost_round=rounds)
        y_pred = model.predict(xgb.DMatrix(x_test))
        model.save_model('testXGboostClass.model')  # 保存训练模型

        yprob = np.argmax(y_pred, axis=1)  # return the index of the biggest pro

        predictions = [round(value) for value in yprob]

        # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        #data_test = xgb.DMatrix(x_test, y_test)

        #pred = model.predict(data_test)


        #print(pred)
        #acc = accuracy_score(y_test,pred)

        #print('xgboost ', acc)