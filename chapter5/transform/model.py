#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 
# @desc :

import tensorflow as tf
import numpy as np
from .data_load import load_vocab

from .utils import convert_idx_to_token_tensor
from .modules import get_token_embeddings,positional_encoding,multihead_attention,ff,label_smoothing,noam_scheme
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class Transformer:

    def __init__(self,hp):
        self.hp = hp
        self.token2idx,self.idx2token = load_vocab(hp.vocab)
        self.embeddings = get_token_embeddings(self.hp.vocab_size,self.hp_d_model,zero_pad = True)


    def encode(self,xs,training = True):
        logging.INFO("encode .....")
        with tf.compat.v1.variable_scope("encoder",reuse=tf.compat.v1.AUTO_REUSE):
            x,seqlens,sents1 = xs
            src_masks = tf.math.equal(x,0)

            enc = tf.compat.v1.nn.embedding_lookup(self.embeddings,x)
            enc *= self.hp.d_model ** 0.5 # scale

            enc += positional_encoding(enc,self.hp.maxlen1)
            enc = tf.compat.v1.layers.dropout(enc,self.dropout_rate,training=training)

            for i in range(self.hp.num_blocks):
                with tf.compat.v1.variable_scope("num_blocks_{}".format(i),reuse=tf.compat.v1.AUTO_REUSE):
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              cavsality=False)
                    enc = ff(enc,num_units=[self.hp.d_ff,self.hp.d_model])
        memory = enc
        logging.INFO("Done")
        return memory,sents1,src_masks

    def decode(self,ys,memory,src_masks,training =True):
        logging.INFO("decode......")
        with tf.compat.v1.variable_scope("decoder",reuse=tf.compat.v1.AUTO_REUSE):
            decoder_inputs,y,seqlens,sents2 = ys

            tgt_masks = tf.math.equal(decoder_inputs,0)

            dec = tf.compat.v1.nn.embedding_lookup(self.embeddings,decoder_inputs)
            dec *= self.hp.d_model ** 0.5

            dec += positional_encoding(dec,self.hp.maxlen2)
            dec = tf.compat.v1.layers.dropout(dec,self.hp.dropout_rate,training=training)

            for i in range(self.hp.num_blocks):
                with tf.compat.v1.variable_scope("num_blocks_{}".format(i),reuse=tf.compat.v1.AUTO_REUSE):
                    dec = multihead_attention(queries= dec,
                                              keys = dec,
                                              values= dec,
                                              key_masks=tgt_masks,
                                              num_heads= self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training = training,
                                              cavsality= True,
                                              scope = "sele_attention")

                    dec = multihead_attention(queries=dec,
                                              keys = memory,
                                              values= memory,
                                              key_masks = src_masks,
                                              num_heads = self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              cavsality = False,
                                              scope="vanilla_attention")
                    dec = ff(dec,num_units=[self.hp.d_ff,self.hp.d_model])

        weights = tf.transpose(self.embeddings)
        logits = tf.einsum('ntd,dk->ntk',dec,weights)
        y_hat = tf.compat.v1.to_int32(tf.argmax(logits,axis=-1))
        logging.INFO("Done")

        return logits,y_hat,y,sents2


    def train(self,xs,ys):
        logging.INFO("train begin...")
        memory,sents1,src_marks = self.encode(xs)
        logits,preds,y,sents2 = self.decode(ys,memory,src_marks)

        y_ = label_smoothing(tf.one_hot(y,depth=self.hp.vocab_size))
        ce = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y_)
        nonpadding = tf.compat.v1.to_float(tf.not_equal(y,self.token2idx["<pad>"]))
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        global_step = tf.compat.v1.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr,global_step,self.hp.warmup_steps)
        optimizer = tf.compat.v1.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss,global_step = global_step)

        tf.compat.v1.summary.scalar('lr',lr)
        tf.compat.v1.summary.scalar("loss",loss)
        tf.compat.v1.summary.scalar("global_step",global_step)

        summaries = tf.compat.v1.summary.merge_all()

        return loss, train_op, global_step,summaries


    def eval(self,xs,ys):
        logging.INFO("eval begin....")
        decoder_inputs, y, y_seqlen, sents2 = ys
        decoder_inputs = tf.compat.v1.ones((tf.shape(xs[0])[0], 1), tf.compat.v1.int32) * self.token2idx["<s>"]

        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1, src_masks = self.encode(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.maxlen2)):
            logits, y_hat, y, sents2 = self.decode(ys, memory, src_masks, False)
            """ .all() """
            if np.array(tf.compat.v1.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]).all(): break

            _decoder_inputs = tf.compat.v1.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        n = tf.compat.v1.random_uniform((), 0, tf.compat.v1.shape(y_hat)[0] - 1, tf.int32)
        sent1 = sents1[n]
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        sent2 = sents2[n]

        tf.compat.v1.summary.text("sent1", sent1)
        tf.compat.v1.summary.text("pred", pred)
        tf.compat.v1.summary.text("sent2", sent2)

        # tf.compat.v1.disable_eager_execution()
        summaries = tf.compat.v1.summary.merge_all()

        logging.INFO("Done!")
        return y_hat, summaries
