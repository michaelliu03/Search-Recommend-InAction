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


##
#
#
# ###
def scaled_dot_product_attention(Q,K,V,key_masks,
                              cavsality=False,
                              dropout_rate=0.,
                              training =True,
                              scope = "scaled_dot_product_attention"):

    with tf.compat.v1.variable_scope(scope,reuse=tf.compat.v1.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        outputs = tf.matmul(Q,tf.transpose(K,[0,2,1]))

        outputs /= d_k ** 0.5

        outputs = mask(outputs,key_masks= key_masks,type = "type")

        if cavsality:
            outputs = mask(outputs,type = "future")

        outputs = tf.nn.softmax(outputs)
        attention =  tf.transpose(outputs,[0,2,1])
        tf.summary.image("attention",tf.expand_dims(attention[:1],-1))

        outputs = tf.compat.v1.layers.dropout(outputs,rate=dropout_rate,training=training)

        outputs = tf.matmul(outputs,V)

    return outputs


def mask(inputs,key_masks = None,type = None):
    padding_num = -2 ** 32 +1
    if type in ("k","key","keys"):
        key_masks = tf.compat.v1.to_float(key_masks)
        key_masks = tf.tile(key_masks,[tf.shape(inputs)[0] // tf.shape(key_masks)[0],1])
        key_masks = tf.expand_dims(key_masks,1)
        outputs = inputs + key_masks * padding_num
    elif type in ("f","future","right"):
        diag_vals = tf.ones_like(inputs[0,:,:])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        future_masks = tf.tile(tf.expand_dims(tril,0),[tf.shape(inputs)[0],1,1])
        paddings = tf.ones_like(future_masks) * padding_num
        outputs = tf.where(tf.equal(future_masks,0),paddings,inputs)

    else:
        print("Check if you entered type correctly!")

    return outputs


def multihead_attention(queries,keys,values,key_masks,
                        num_heads =8,
                        dropout_rate =0,
                        training = True,
                        cavsality = False,
                        scope = "nultihead_attention"):
    d_model = queries.get_shape().as_list()[-1]
    with tf.compat.v1.variable_scope(scope,reuse=tf.compat.v1.AUTO_REUSE):
        Q = tf.compat.v1.layers.dense(queries, d_model, use_bias=True)  # (N, T_q, d_model)
        K = tf.compat.v1.layers.dense(keys, d_model, use_bias=True)  # (N, T_k, d_model)
        V = tf.compat.v1.layers.dense(values, d_model, use_bias=True)  # (N, T_k, d_model)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, key_masks, cavsality, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = ln(outputs)

    return outputs




def ff(inputs,num_units,scope="positionwise_feedforward"):

    with tf.compat.v1.variable_scope(scope,reuse=tf.compat.v1.AUTO_REUSE):

        outputs = tf.compat.v1.layers.dense(inputs,num_units[0],activation=tf.nn.relu)

        outputs = tf.compat.v1.layers.dense(outputs,num_units[1])

        #Residual connection
        outputs += inputs

        # Normalize
        outputs = ln(outputs)

    return outputs