#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:dssm_model.py
# @Author: Michael.liu
# @Date:2020/6/11 15:08
# @Desc: this code is ....

import tensorflow as tf
class DnnDssmModel(object):
    def __init__(self,config ,init_size,batch_size=None,samples = None,is_trainging=True):
        self.config = config
        self.batch_size = config["batch_size"]
        self.samples = config["neg_sample"] +1
        self.init_size = init_size

        if not is_trainging:
            self.batch_size =batch_size
            self.samples = samples

        self.query = tf.placeholder(tf.int32,[self.batch_size,None],name="query")
        self.sim_query = tf.placeholder(tf.int32,[self.batch_size * self.samples,None],name="sim_query")

        # create graph
        self.model_structure()

        # init saver
        self.init_saver()

    def model_structure(self):

        with tf.name_scope("DNN"):
            query_embedded = self.query
            sim_query_embedded = self.sim_query
            init_hidden_size = self.init_size

            for idx,hidden_size in enumerate(self.config["hidden_sizes"]):
                query_embedded = self.hidden_layer(query_embedded,init_hidden_size,hidden_size,"dnn" +str(idx))
                sim_query_embedded = self.hidden_layer(sim_query_embedded,init_hidden_size,hidden_size,"dnn"+str(idx),True)
                init_hidden_size = hidden_size
            query_final_output = query_embedded
            sim_query_final_output = sim_query_embedded

        with tf.name_scope("reshape_sim"):
            split_sim_query_final_output = tf.split(sim_query_final_output,
                                                    [self.samples] * self.batch_size,
                                                    axis = 0)
            expand_concat_sim_query = [tf.expand_dims(tensor,0) for tensor in split_sim_query_final_output]
            reshape_sim_query_final_output = tf.concat(expand_concat_sim_query,axis=0)

        with tf.name_scope("cosin_similarity"):
            expand_query_final_output = tf.expand_dims(query_final_output,1)
            query_norm = tf.sqrt(tf.reduce_sum(tf.square(expand_query_final_output),-1))
            sim_query_norm = tf.sqrt(tf.reduce_sum(tf.square(reshape_sim_query_final_output),-1))
            dot = tf.reduce_sum(tf.multiply(expand_query_final_output,reshape_sim_query_final_output),axis=-1)

            norm = query_norm * sim_query_norm
            self.similarity = tf.div(dot,norm,name = "similarity")
            self.predictions = tf.argmax(self.similarity,-1,name= "predictions")

        with tf.name_scope("loss"):
            if self.config["neg_samples"] == 1:
                pos_similarity = tf.reshape(tf.slice(self.similarity, [0, 0], [self.batch_size, 1]),
                                            [self.batch_size])
                neg_similarity = tf.reshape(tf.slice(self.similarity,
                                                     [0, 1],
                                                     [self.batch_size, self.config["neg_samples"]]),
                                            [self.batch_size])
                distance = self.config["margin"] - pos_similarity + neg_similarity
                zeros = tf.zeros_like(distance, dtype=tf.float32)
                cond = (distance >= zeros)
                losses = tf.where(cond, distance, zeros)
                self.loss = tf.reduce_mean(losses, name="loss")
            else:
                pos_similarity = tf.exp(tf.reshape(tf.slice(self.similarity, [0, 0], [self.batch_size, 1]),
                                                   [self.batch_size]))
                neg_similarity = tf.exp(
                    tf.slice(self.similarity, [0, 1], [self.batch_size, self.config["neg_samples"]])
                )
                norm_seg_similarity = tf.reduce_sum(neg_similarity, axis=-1)
                pos_prob = tf.div(pos_similarity, norm_seg_similarity)
                self.loss = tf.reduce_mean(-tf.log(pos_prob), name="loss")

        with tf.name_scope("train_op"):
            # 定义优化器
            optimizer = self.get_optimizer()

            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            # 对梯度进行梯度截断
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_grad_norm"])
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

    @staticmethod
    def hidden_layer(inputs, init_hidden_size, final_hidden_size, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            w = tf.get_variable("w", shape=[init_hidden_size, final_hidden_size],
                                initializer=tf.glorot_normal_initializer())
            b = tf.get_variable("b", shape=[final_hidden_size], initializer=tf.glorot_normal_initializer())
            outputs = tf.nn.xw_plus_b(inputs, w, b)
            return outputs

    def get_optimizer(self):
        """
        获得优化器
        :return:
        """
        optimizer = None
        if self.config["optimization"] == "adam":
            optimizer = tf.train.AdamOptimizer(self.config["learning_rate"])
        if self.config["optimization"] == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.config["learning_rate"])
        if self.config["optimization"] == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.config["learning_rate"])
        return optimizer

    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables())


    def train(self,sess,batch,dropout_prob):
        feed_dict = {
            self.query:batch["query"],
            self.sim_query:batch["sim"],
            self.query_length:batch["query_length"],
            self.sim_length: batch["sim_length"],
            self.keep_prob:dropout_prob
        }

        _,loss,predictions = sess.run([self.train_op,self.loss,self.predictions],feed_dict=feed_dict)

        return loss,predictions

    def eval(self,sess,batch):
        feed_dict = {self.query:batch["query"],
                     self.sim_query:batch["sim"],
                     self.query_length:batch["query_length"],
                     self.sim_length:batch["sim_length"],
                     self.keep_prob:1.0}
        loss,predictions = sess.run([self.loss,self.predictions],feed_dict=feed_dict)
        return loss,predictions

    def infer(self,sess,batch):
        feed_dict = {
            self.query:batch["query"],
            self.sim_query:batch["sim"],
            self.query_length:batch["query_length"],
            self.sim_length:batch["sim_length"],
            self.keep_prob:1.0
        }

        predict = sess.run(self.predictions,feed_dict =feed_dict)

        return predict

