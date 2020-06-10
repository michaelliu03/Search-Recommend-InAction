#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:Model.py
# @Author: Michael.liu
# @Date:2020/6/9 10:56
# @Desc: this code is ....
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import pickle as pkl
import tensorflow as tf

dtype = tf.float32
class Model:
    def __init__(self):
        self.sess = None
        self.X = None
        self.y = None
        self.layer_keeps = None
        self.vars = None
        self.keep_prob_train = None
        self.keep_prob_test = None

    def run(self,fetches, X =None,y=None,mode='train'):
        feed_dic = {}
        if type(self.X) is list:
            for i in range(len(X)):
                feed_dic[self.X[i]] = X[i]
        else:
            feed_dic[self.X] = X
        if y is not None:
            feed_dic[self.y] = y
        if self.layer_keeps is not None:
            if mode =='train':
                feed_dic[self.layer_keeps] = self.keep_prob_train
            elif mode == 'test':
                feed_dic[self.layer_keeps] = self.keep_prob_test
        return self.sess.run(fetches,feed_dic)

    def dump(self,model_path):
        var_map = {}
        for name , var  in self.vars.iteritem():
            var_map[name] = self.run(var)
        pkl.dump(var_map,open(model_path,'wb'))
        print('model dumped at',model_path)
