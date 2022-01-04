#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 
# @desc :
import numpy as np
import tensorflow as tf

def ln(inputs,epsilon = 1e-8,scope="ln"):

    with tf.compat.v1.variable_scope(scope,reuse=tf.compat.v1.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean,variance = tf.nn.moments(inputs,[-1],keepdims=True)
        beta = tf.compat.v1.get_variable("beta",params_shape,initializer=tf.zeros_initializer())
        gamma = tf.compat.v1.get_variable("gamma",params_shape,initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def get_token_embeddings(vocab_size,num_units,zero_pad=True):

    with tf.compat.v1.variable_scope("shared_weight_matrix"):
        embedding = tf.compat.v1.get_variable('weight_mat',
                                              dtype=tf.float32,
                                              shape=(vocab_size,num_units),
                                              initializer= tf.compat.v1.initializers.glorot_uniform())
        if zero_pad:
           embedding = tf.compat.v1.concat((tf.zeros(shape=[1,num_units]),embedding[1:,:]),0)

    return embedding

def scaled_dot_product_attion(Q,K,V,key_masks,cavsality=False,dropout_rate=0.,training =True,scope = "scaled_dot_product_attention"):
    with tf.compat.v1.variable_scope(scope,reuse=tf.compat.v1.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        outputs = tf.matmul(Q,tf.transpose(K,[0,2,1]))




