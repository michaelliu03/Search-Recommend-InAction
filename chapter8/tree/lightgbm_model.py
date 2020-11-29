#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:lightgbm_model.py
# @Author: Michael.liu
# @Date:2020/7/6 17:35
# @Desc: this code is ....
import lightgbm as lgb

class lgbt():
    LGB_EXEC = ''

    def __init__(self, LGB_exec='/home/challenger/LightGBM/lightgbm',):
        self.LGB_EXEC = LGB_exec
        self.train_file = self.config["dt_train_file"]
        self.vali_file = self.config["dt_vali_file"]
        self.test_file = self.config["dt_test_file"]

    def train_model(self, train_data_path, test_data_path, params):
        train_data = lgb.Dataset(train_data_path)
        test_data = lgb.Dataset(test_data_path)

        gbm = lgb.train(params=params, train_set=train_data, valid_sets=[train_data, test_data])
        print('Save model...')
        gbm.save_model('save_model')  # 训练后保存模型到文件


