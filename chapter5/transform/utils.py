#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 
# @desc :

import tensorflow as tf
import json
import os,re
import logging

logging.basicConfig(level= logging.INFO)

def calc_num_batches(total_num,batch_size):

    return total_num // batch_size + int(total_num % batch_size != 0)

def convert_idx_to_token_tensor(inputs,idx2token):

    def my_func(inputs):
        return " ".join(idx2token[elem] for elem in inputs)

    return tf.compat.v1.py_func(my_func,[inputs],tf.string)


def postprocess(hypotheses, idx2token):
    '''Processes translation outputs.
    hypotheses: list of encoded predictions
    idx2token: dictionary

    Returns
    processed hypotheses
    '''
    _hypotheses = []
    for h in hypotheses:
        sent = "".join(idx2token[idx] for idx in h)  # 合并peice 成vocab
        sent = sent.split("</s>")[0].strip()  # 去掉结尾符
        sent = sent.replace("▁", " ")  # remove bpe symbols   # 单词间分隔符用空格代替

        _hypotheses.append(sent.strip())
    return _hypotheses

def save_hparams(hparams,path):
    if not os.path.exists(path): os.makedirs(path)
    hp = json.dumps(vars(hparams))
    with open(os.path.join(path,"hparams"),'w') as fout:
        fout.write(hp)

def save_variable_specs(fpath):

    def _get_size(shp):
        size = 1
        for d in range(len(shp)):
            size *=shp[d]
        return size

    params, num_params = [], 0

    for v in tf.compat.v1.global_variables():
        params.append("{}==={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    print("num_params: ", num_params)
    with open(fpath, 'w') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))
    logging.info("Variables info has been saved.")

def get_hypotheses(num_batches,num_samples,sess,tensor,dict):
    hypotheses = []
    for _ in range(num_batches):
        h = sess.run(tensor)
        hypotheses.extend(h.tolist())
    hypotheses = postprocess(hypotheses,dict)

    return hypotheses[:num_samples]

def load_hparams(parser,path):
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    d = open(os.path.join(path,"hparams"),'r').read()
    flag2val = json.loads(d)
    for f,v in flag2val.items():
        parser.f = v

def save_variable_specs(fpath):

    def _get_size(shp):

        size = 1
        for d in range(len(shp)):
            size *= shp[d]
        return size

    params,num_params =[],0
    for v in tf.compat.v1.global_variables():
        params.append("{}==={}".format(v.name,v.shape))
        num_params += _get_size(v.shape)
    print("num_params: ",num_params)
    with open(fpath,'w') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))
    logging.info("Variables info has been saved.")


def get_hypotheses(num_batches,num_samples,sess,tensor,dict):
    hypotheses = []
    for _ in range(num_batches):
        h = sess.run(tensor)
        hypotheses.extend(h.tolist())

    return hypotheses[:num_samples]




