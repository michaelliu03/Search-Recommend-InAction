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

def load_data(fpath1,fpath2,maxlen1,maxlen2):
    sents1,sents2 = [],[]
    with open(fpath1,'r',encoding='utf-8',errors='ignore') as f1,open(fpath2,'r',encoding='utf-8',errors = 'ignore') as f2:
        for sents1,sents2 in zip(f1,f2):
            if len(sents1.split()) + 1 > maxlen1:continue
            if len(sents2.split()) + 1 > maxlen2: continue

            sents1.append(sents1.strip())
            sents2.append(sents2.strip())
    return sents1,sents2


def encode(inp,type,dict):
   inp_str = inp.decode("utf-8")
   if type == "x":
       tokens = inp_str.split() + ["</s>"]
   else:
       tokens = ["<s>"]  + inp_str.split() + ["</s>"]
   x = [dict.get(t,dict["<unk>"]) for t in tokens]
   return x

def generator_fn(sents1, sents2, vocab_fpath):
    '''Generates training / evaluation data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.    bpe.vocab  word2id

    yields
    xs: tuple of
        x: list of source token ids in a sent
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) sentence
    labels: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs
        y: list of target token ids in a sent
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    '''
    token2idx, _ = load_vocab(vocab_fpath)
    for sent1, sent2 in zip(sents1, sents2):
        x = encode(sent1, "x", token2idx)
        y = encode(sent2, "y", token2idx)
        decoder_input, y = y[:-1], y[1:]  # 前n-1 个为decode的输入， 后n - 1为训练集标签

        x_seqlen, y_seqlen = len(x), len(y)
        yield (x, x_seqlen, sent1), (decoder_input, y, y_seqlen, sent2)

def input_fn(sents1,sents2,vocab_fpath,batch_size,shuffle =False):
    shapes = (([None],(),()),
              ([None],[None],(),()))
    types = ((tf.compat.v1.int32,tf.compat.v1.int32,tf.compat.v1.String),
             (tf.compat.v1.int32,tf.compat.v1.int32,tf.compat.v1.int32,tf.compat.v1.string))

    paddings = ((0,0,''),(0,0,0,''))

    dataset = tf.compat.v1.data.Dataset.from_generator(
               generator_fn,
               output_shapes=shapes,
               output_types=types,
               args=(sents1, sents2, vocab_fpath))

    if shuffle:
        dataset = dataset.shuffle(128 * batch_size)

    dataset = dataset.repeat()
    dataset = dataset.padded_batch(batch_size,shapes,paddings).prefetch(1)

    return dataset


def get_batch(fpath1,fpath2,maxlen1,maxlen2,vocab_fpth,batch_size,shuffle=False):

    sents1,sents2 = load_data(fpath1,fpath2,maxlen1,maxlen2)
    print("sents1:")
    print(sents1[:2])
    print("sents2:")
    print(sents2[:2])
    batches = input_fn(sents1, sents2, vocab_fpth, batch_size, shuffle=shuffle)  # dataset 加载数据
    num_batches = calc_num_batches(len(sents1), batch_size)
    print("num_batches: ", num_batches)
    return batches, num_batches, len(sents1)


