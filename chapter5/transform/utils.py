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
