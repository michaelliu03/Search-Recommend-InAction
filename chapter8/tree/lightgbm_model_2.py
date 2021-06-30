#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 
# @desc :

import argparse
import os
import sys
import types
import unittest

import lightgbm as lgb
import config

import  pandas as pd
from sklearn.metrics import f1_score,accuracy_score,recall_score,log_loss,roc_curve,auc
def evalue(y_pred,x_test,y_test,label):
    '''
    :param y_pred:
    :param x_test:
    :param y_test:
    :param label:
    f1_score,accuracy_score,recall_score,log_loss,auc
    :return:
    '''

    f1 = f1_score(y_test,y_pred)
    acc = accuracy_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    log = log_loss(y_test,y_pred)
    fpr,tpr,threshold = roc_curve(y_test,y_pred)
    auc_value = auc(fpr,tpr)

    display = '''
        evluation of {label} is :
        f1_score = {f1}
        accuracy = {acc}
        recall = {recall}
        log_loss = {log}
        auc = {auc_value}
    '''
    print(display.format(label=label,f1=f1,acc=acc,recall=recall,log=log,auc_value=auc_value))

class lgbt():
    LGB_EXEC = ''

    def __init__(self, LGB_exec='lgb_model_path'):
        self.LGB_EXEC = LGB_exec



    def train_model(self, train_data_path, test_data_path, params):
        print(train_data_path)
        #head = 'type,tfidf_cnt,tfidf&tar_cnt,tech_cnt,tech&tar_cnt,agentList_cnt,problem_cnt,problem&tar_cnt,func_cnt,func&tar_cnt,applicantFirst,target'.split(',')
        head = 'similar,prod_img_cnt,rec_prod_img_cnt,ai_score,rec_ai_score,cate2,cate3,cate4,is_free_ship,fav_count,rfx_count,promo_flag,pt_score,review_pv,main_ic,rec_ic,click_flag'.split(',')
        #head = head.split(",")
        print(head)

        train_data_pd = pd.read_csv(train_data_path,sep=',',header=0,encoding='utf8')
        test_data_pd = pd.read_csv(test_data_path,sep=',',header=0,encoding='utf8')

        x_train = train_data_pd[head[:-2]]
        y_train = train_data_pd[head[-1]]

        x_test = test_data_pd[head[:-2]]
        y_test = test_data_pd[head[-1]]

        # x_train = train_data_pd[head_list[:-2]]
        # y_train = train_data_pd[head_list[-1]]
        # #
        # x_test = test_data_pd[head_list[:-2]]
        # y_test = test_data_pd[head_list[-1]]

        train_data = lgb.Dataset(x_train,y_train)
        test_data = lgb.Dataset(x_test,y_test)


        # command = '{} {} {}'.format(LGB_EXEC,
        #                            ' '.join(unknown_args),
        #                            ' '.join('{}={}'.format(x, format_param_val(y)) for x, y in params.items()))
        # print("训练任务命令: %s" % command)
        # os.chdir(task_path)

        # os.system(command)

        gbm = lgb.train(params=params, train_set=train_data, valid_sets=[train_data, test_data])

        print('Save model...')
        y_pred= gbm.predict(x_test)

        y_pred = [ 1 if i>0.5 else 0 for i in y_pred]


        print(y_pred)
        evalue(y_pred, x_test, y_test, 'lgb')
        # gbm.save_model('save_model')  # 训练后保存模型到文件


def main():
    lgb = lgbt()

    print(config.params)
    lgb.train_model(config.params['train_data'], config.params['test_data'], config.params)



import time
s = time.time()
main()
e = time.time()

print( (e-s))