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
from .modules import get_token_embeddings
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


        logging.INFO("Done")

    def decode(self,ys,memory,src_masks,training =True):
        logging.INFO("decode......")

        logging.INFO("Done")

    def train(self,xs,ys):
        logging.INFO("train begin...")
        memory,sents1,src_marks = self.encode(xs)
        logits,preds,y,sents2 = self.decode(ys,memory,src_marks)
       #y_ =

        logging.INFO("Done!")