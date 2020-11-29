#!/usr/bin/env python
# -*-coding:utf-8-*-
# @File:dssm_model.py
# @Author: Michael.liu
# @Date:2020/6/11 15:08
# @Desc: this code is ....

import tensorflow as tf
from .helper import *

TRIGRAM_D = 100
# negative sample
NEG = 4
# query batch size
query_BS = 100
# batch size
BS = query_BS * NEG


class DssmModel(object):


    def __init__(self, config):

        self.config = config
        # self.batch_size = config["batch_size"]
        self.vocab_map = DssmData.load_vocab(self.config["vocab_path"])
        self.nwords = len(self.vocab_map)

        #self.use_stack_rnn= self.config["use_stack_rnn"]

        # if self.use_stack_rnn == True:
        #     self.hidden_size_rnn= self.config["hidden_size_rnn"]
        # self.optimization = self.config["optimization"]
        # self.max_seq_len = self.config["max_seq_len"]

        self.use_stack_rnn = self.config["use_stack_rnn"]

        # if self.use_stack_rnn == True:
        self.hidden_size_rnn = self.config["hidden_size_rnn"]
        self.optimization = self.config["optimization"]
        self.max_seq_len = self.config["max_seq_len"]


        # create graph
        self.model_structure()

        # init saver
        self.init_saver()

    def model_structure(self):

        with tf.name_scope('input'):
            # 预测时只用输入query即可，将其embedding为向量。
            print("input")

        with tf.name_scope('word_embeddings_layer'):
            _word_embedding = tf.get_variable(name="word_embedding_arr", dtype=tf.float32,
                                              shape=[self.nwords, TRIGRAM_D])
            query_embed = tf.nn.embedding_lookup(_word_embedding, self.query_batch, name='query_batch_embed')
            doc_pos_embed = tf.nn.embedding_lookup(_word_embedding, self.doc_pos_batch, name='doc_positive_embed')
            doc_neg_embed = tf.nn.embedding_lookup(_word_embedding, self.doc_neg_batch, name='doc_negative_embed')

        with tf.name_scope('RNN'):
            # Abandon bag of words, use GRU, you can use stacked gru
            # query_l1 = add_layer(query_batch, TRIGRAM_D, L1_N, activation_function=None)  # tf.nn.relu()
            # doc_positive_l1 = add_layer(doc_positive_batch, TRIGRAM_D, L1_N, activation_function=None)
            # doc_negative_l1 = add_layer(doc_negative_batch, TRIGRAM_D, L1_N, activation_function=None)
            if self.use_stack_rnn == "True":
                cell_fw = tf.contrib.rnn.GRUCell(self.hidden_size_rnn, reuse=tf.AUTO_REUSE)
                stacked_gru_fw = tf.contrib.rnn.MultiRNNCell([cell_fw], state_is_tuple=True)
                cell_bw = tf.contrib.rnn.GRUCell(self.hidden_size_rnn, reuse=tf.AUTO_REUSE)
                stacked_gru_bw = tf.contrib.rnn.MultiRNNCell([cell_fw], state_is_tuple=True)
                (output_fw, output_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(stacked_gru_fw, stacked_gru_bw)
                # not ready, to be continue ...

            if self.use_stack_rnn == "False":
                cell_fw = tf.contrib.rnn.GRUCell(self.hidden_size_rnn, reuse=tf.AUTO_REUSE)
                cell_bw = tf.contrib.rnn.GRUCell(self.hidden_size_rnn, reuse=tf.AUTO_REUSE)
                # query
                (_, _), (query_output_fw, query_output_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                             query_embed,
                                                                                             sequence_length=self.query_seq_length,
                                                                                             dtype=tf.float32)
                query_rnn_output = tf.concat([query_output_fw, query_output_bw], axis=-1)
                query_rnn_output = tf.nn.dropout(query_rnn_output, self.drop_out_prob)
                # doc_pos
                (_, _), (doc_pos_output_fw, doc_pos_output_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                                 doc_pos_embed,
                                                                                                 sequence_length=self.pos_seq_length,
                                                                                                 dtype=tf.float32)
                doc_pos_rnn_output = tf.concat([doc_pos_output_fw, doc_pos_output_bw], axis=-1)
                doc_pos_rnn_output = tf.nn.dropout(doc_pos_rnn_output, self.drop_out_prob)

                # doc_neg
                (_, _), (doc_neg_output_fw, doc_neg_output_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                                 doc_neg_embed,
                                                                                                 sequence_length=self.neg_seq_length,
                                                                                                 dtype=tf.float32)
                doc_neg_rnn_output = tf.concat([doc_neg_output_fw, doc_neg_output_bw], axis=-1)
                doc_neg_rnn_output = tf.nn.dropout(doc_neg_rnn_output, self.drop_out_prob)

        with tf.name_scope('merge_negative_doc'):
            # 合并负样本，tile可选择是否扩展负样本。
            # doc_y = tf.tile(doc_positive_y, [1, 1])
            doc_y = tf.tile(doc_pos_rnn_output, [1, 1])

            for i in range(NEG):
                for j in range(query_BS):
                    # slice(input_, begin, size)切片API
                    # doc_y = tf.concat([doc_y, tf.slice(doc_negative_y, [j * NEG + i, 0], [1, -1])], 0)
                    doc_y = tf.concat([doc_y, tf.slice(doc_neg_rnn_output, [j * NEG + i, 0], [1, -1])], 0)

        with tf.name_scope('cosine_similarity'):
            # Cosine similarity
            # query_norm = sqrt(sum(each x^2))
            query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_rnn_output), 1, True)), [NEG + 1, 1])
            # doc_norm = sqrt(sum(each x^2))
            doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))

            prod = tf.reduce_sum(tf.multiply(tf.tile(query_rnn_output, [NEG + 1, 1]), doc_y), 1, True)
            norm_prod = tf.multiply(query_norm, doc_norm)

            # cos_sim_raw = query * doc / (||query|| * ||doc||)
            cos_sim_raw = tf.truediv(prod, norm_prod)
            # gamma = 20
            cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, query_BS])) * 20

        with tf.name_scope('loss'):
            # Train Loss
            # 转化为softmax概率矩阵。
            prob = tf.nn.softmax(cos_sim)
            # 只取第一列，即正样本列概率。
            hit_prob = tf.slice(prob, [0, 0], [-1, 1])
            self.loss = -tf.reduce_sum(tf.log(hit_prob))
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('train_op'):
            # Optimizer
            self.train_step = tf.train.AdamOptimizer(conf.learning_rate).minimize(self.loss)

        with tf.name_scope('vali'):
            self.average_loss = tf.placeholder(tf.float32)
            self.loss_summary = tf.summary.scalar('average_loss', self.average_loss)

        with tf.name_scope('Train'):
            self.train_average_loss = tf.placeholder(tf.float32)
            self.train_loss_summary = tf.summary.scalar('train_average_loss', self.train_average_loss)

    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar('sttdev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))
            tf.summary.histogram(name, var)

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

    def gen_data(self, data_map, batch_id):
        query_in = data_map['query'][batch_id * query_BS:(batch_id + 1) * query_BS]
        query_len = data_map['query_len'][batch_id * query_BS:(batch_id + 1) * query_BS]
        doc_positive_in = data_map['doc_pos'][batch_id * query_BS:(batch_id + 1) * query_BS]
        doc_positive_len = data_map['doc_pos_len'][batch_id * query_BS:(batch_id + 1) * query_BS]
        doc_negative_in = data_map['doc_neg'][batch_id * query_BS * NEG:(batch_id + 1) * query_BS * NEG]
        doc_negative_len = data_map['doc_neg_len'][batch_id * query_BS * NEG:(batch_id + 1) * query_BS * NEG]

        # query_in, doc_positive_in, doc_negative_in = pull_all(query_in, doc_positive_in, doc_negative_in)
        return query_in, doc_positive_in, doc_negative_in, query_len, doc_positive_len, doc_negative_len

    def feed_dict(self, on_training, data_set, batch_id, drop_prob):
        query_in, doc_positive_in, doc_negative_in, query_seq_len, pos_seq_len, neg_seq_len = self.gen_data(data_set,
                                                                                                            batch_id)
        query_len = len(query_in)
        query_seq_len = [self.config["max_seq_len"]] * query_len
        pos_seq_len = [self.config["max_seq_len"]] * query_len
        neg_seq_len = [self.config["max_seq_len"]] * query_len * NEG
        return {self.query_batch: query_in, self.doc_pos_batch: doc_positive_in, self.doc_neg_batch: doc_negative_in,
                self.on_train: on_training, self.drop_out_prob: drop_prob, self.query_seq_length: query_seq_len,
                self.neg_seq_length: neg_seq_len, self.pos_seq_length: pos_seq_len}

    def train(self, sess, data_train, batch_id):
        _, loss = sess.run([self.train_step, self.loss], feed_dict=self.feed_dict(False, data_train, batch_id, 0.5))

        return loss


