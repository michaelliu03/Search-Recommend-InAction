#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:FM_Model.py
# @Author: Michael.liu
# @Date:2020/6/9 18:04
# @Desc: this code is ....
from .Model import *
from .helper_tf import *

class FM_Model(Model):
    def __init__(self,
                 input_dim =None,
                 output_dim =1,
                 factor_order=10,
                 init_path=None,
                 opt_algo='gd',
                 learning_rate=1e-2,
                 l2_w=0,
                 l2_v=0,
                 random_seed =None):
        Model.__init__(self)
        init_vars = [('w',[input_dim,output_dim],'xavier',dtype),
                     ('v',[input_dim,factor_order],'xavier',dtype),
                     ('b',[output_dim],'zero',dtype)]
        self.graph = tf.Graph()
        with self.graph.as_default():
            if random_seed is not None:
                tf.set_random_seed(random_seed)
            self.X = tf.sparse_placeholder(dtype)
            self.y = tf.placeholder(dtype)
            self.vars = init_var_map(init_vars,init_path)

            w = self.vars['w']
            v = self.vars['v']
            b = self.vars['b']

            X_square = tf.SparseTensor(self.X.indices,tf.square(self.X.values),tf.to_int64(tf.shape(self.X)))
            xv = tf.square(tf.sparse_tensor_dense_matmul(self.X,v))
            p = 0.5 * tf.reshape(tf.reduce_sum(xv-tf.sparse_tensor_dense_matmul(X_square,X_square(v)),1),[-1,output_dim])
            xw =tf.sparse_tensor_dense_matmul(self.X,w)
            logits = tf.reshape(xw+b+p,[-1])
            self.y_prob = tf.sigmoid(logits)

            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=self.y)+l2_w*tf.nn.l2_loss(xw)+l2_v*tf.nn.l2_loss(xv))
            self.optimizer = get_optimizer(opt_algo,learning_rate,self.loss)

            config =tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
            # config.gpu_options.allow_growth = True
            # self.sess = tf.Session(config=config)
            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

            # 初始化图里的参数
            # print
            tf.global_variables_initializer().run(session=self.sess)
            saver = tf.train.Saver()
            saver.save(self.sess, "Model/FM_model.ckpt")



