#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:lr_tf_train.py
# @Author: Michael.liu
# @Date:2020/6/9 13:22
# @Desc: this code is ....
import numpy as np
from sklearn.metrics import roc_auc_score
import progressbar
import pickle as pkl
from .helper_tf import *
from .LR_Model import *

FIELD_SIZES, FIELD_OFFSETS = read_common('/home/jovyan/michael.liu/c12/avazu_ctr/featindex.txt')
print('read end!')

input_dim = 491713
train_file = 'train.txt'
test_file = 'test.txt'

train_data = pkl.load(open('train.pkl','wb'))
test_data = pkl.load(open('test.pkl','wb'))

if train_data[1].ndim > 1:
    print('label must be 1-dim')
    exit(0)
print('read finish')
print('train data size',train_data[0].shape)
print('test data size', test_data[0].shape)

train_size = train_data[0].shape[0]
test_size = test_data[0].shape[0]
num_feas = 26

min_round = 1
num_round = 200
early_stop_round = 5
batch_size = 1024

field_sizes = 26
field_offsets = FIELD_OFFSETS
# 逻辑回归参数设定
lr_params = {
    'input_dim': input_dim,
    'opt_algo': 'gd',
    'learning_rate': 0.1,
    'l2_weight': 0,
    'random_seed': 0
}
print(lr_params)
model = LR(**lr_params)
print("training LR...")

def train(model):
    history_score = []
    for i in range(num_round):
        fetches = [model.optimizer,model.loss]
        if batch_size > 0:
            ls = []
            bar = progressbar.ProgressBar()
            print('[%d]\ttraining...' % i)
            for j in bar(range(int(train_size / batch_size + 1))):
                X_i, y_i = slice(train_data, j * batch_size, batch_size)
                # 训练，run op
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
        # 把预估的结果和真实结果拿出来计算auc
        train_score = roc_auc_score(train_data[1], train_preds)
        test_score = roc_auc_score(test_data[1], test_preds)
            # 输出auc信息
        print('[%d]\tloss (with l2 norm):%f\ttrain-auc: %f\teval-auc: %f' % (i, np.mean(ls), train_score, test_score))
        history_score.append(test_score)
        # early stopping
        if i > min_round and i > early_stop_round:
            if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[
                    -1 * early_stop_round] < 1e-5:
               print('early stop\nbest iteration:\n[%d]\teval-auc: %f' % (
                    np.argmax(history_score), np.max(history_score)))
               break

train(model)

