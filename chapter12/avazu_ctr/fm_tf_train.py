#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:fm_tf_train.py
# @Author: Michael.liu
# @Date:2020/6/10 11:37
# @Desc: this code is ....

import numpy as np
from sklearn.metrics import roc_auc_score
import progressbar
from .helper_tf import *
from .FM_Model import *

train_file = 'train.txt'
test_file = 'test.txt'

FIELD_SIZES, FIELD_OFFSETS = read_common('/home/jovyan/michael.liu/c12/avazu_ctr/featindex.txt')
print('read end!')


input_dim = 491713
train_data = pkl.load(open('train.pkl', 'rb'))
train_data = shuffle(train_data)
test_data = pkl.load(open('test.pkl', 'rb'))

if train_data[1].ndim > 1:
    print('label must be 1-dim')
    exit(0)
print('read finish')
print('train data size:', train_data[0].shape)
print('test data size:', test_data[0].shape)

# 训练集与测试集
train_size = train_data[0].shape[0]
test_size = test_data[0].shape[0]
num_feas = len(FIELD_SIZES)

# 超参数设定
min_round = 1
num_round = 200
early_stop_round = 5
batch_size = 1024

field_sizes = FIELD_SIZES
field_offsets = FIELD_OFFSETS

# FM参数设定
fm_params = {
    'input_dim': input_dim,
    'factor_order': 10,
    'opt_algo': 'gd',
    'learning_rate': 0.1,
    'l2_w': 0,
    'l2_v': 0,
}
print(fm_params)
model = FM(**fm_params)
print("training FM...")

def train(model):
    history_score = []
    for i in range(num_round):
        # 同样是优化器和损失两个op
        fetches = [model.optimizer, model.loss]
        if batch_size > 0:
            ls = []
            bar = progressbar.ProgressBar()
            print('[%d]\ttraining...' % i)
            for j in bar(range(int(train_size / batch_size + 1))):
                X_i, y_i = slice(train_data, j * batch_size, batch_size)
                # 训练
                _, l = model.run(fetches, X_i, y_i)
                ls.append(l)
        elif batch_size == -1:
            X_i, y_i = slice(train_data)
            _, l = model.run(fetches, X_i, y_i)
            ls = [l]
        train_preds = []
        print('[%d]\tevaluating...' % i)
        bar = progressbar.ProgressBar()
        for j in bar(range(int(train_size / 10000 + 1))):
            X_i, _ = slice(train_data, j * 10000, 10000)
            preds = model.run(model.y_prob, X_i, mode='test')
            train_preds.extend(preds)
        test_preds = []
        bar = progressbar.ProgressBar()
        for j in bar(range(int(test_size / 10000 + 1))):
            X_i, _ = slice(test_data, j * 10000, 10000)
            preds = model.run(model.y_prob, X_i, mode='test')
            test_preds.extend(preds)
        train_score = roc_auc_score(train_data[1], train_preds)
        test_score = roc_auc_score(test_data[1], test_preds)
        print('[%d]\tloss (with l2 norm):%f\ttrain-auc: %f\teval-auc: %f' % (i, np.mean(ls), train_score, test_score))
        history_score.append(test_score)
        if i > min_round and i > early_stop_round:
            if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[
                        -1 * early_stop_round] < 1e-5:
                print('early stop\nbest iteration:\n[%d]\teval-auc: %f' % (
                    np.argmax(history_score), np.max(history_score)))
                break

train(model)