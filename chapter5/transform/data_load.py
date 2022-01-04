#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 
# @desc :

import tensorflow as tf

from .utils import calc_num_batches
import tensorflow.compat.v1 as tf
from difflib import get_close_matches
tf.disable_v2_behavior()

def load_vocab(vocab_fpath):
    vocab = [line.split()[0] for line in open(vocab_fpath,'r',encoding='utf-8',errors='ignore').read().splitlines()]
    token2idx = {token:idx for idx,token in enumerate(vocab)}
    idx2token = {idx: token for idx,token in enumerate(vocab)}
    return token2idx,idx2token



