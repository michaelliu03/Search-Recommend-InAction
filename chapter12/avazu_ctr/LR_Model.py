#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:LR_Model.py
# @Author: Michael.liu
# @Date:2020/6/9 11:06
# @Desc: this code is ....
from .Model import *
from .helper_tf import *
import tensorflow as tf

class LR(Model):
    def __init__(self,
                 input_dim=None,
                 output_dim = 1,
                 int_path =None,
                 opt_algo='gd',
                 learning_rate=1e-2,
                 l2_weight=0,
                 random_seed = None):
        Model.__init__(self)
        init_vars = [('w', [input_dim, output_dim], 'xavier', dtype),
                     ('b', [output_dim], 'zero', dtype)]

        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = init_var_map(init_vars,int_path)

            w = self.vars['w']
            b = self.vars['b']

            xw = tf.sparse_tensor_dense_matmul(self.X,w)
            logits = tf.reshape(xw+b,[-1])
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,logits=logits) + l2_weight *tf.nn.l2_loss(xw)
            )
            self.optimizer = get_optimizer(opt_algo,learning_rate,self.loss)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            # 初始化图里的参数
            tf.global_variables_initializer().run(session=self.sess)
            saver = tf.train.Saver()
            saver.save(self.sess,"Model/lr_model.ckpt")