#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @project : books_method
# @File : main.py
# @Time    : 2020/6/24 16:14
# @Author  : Zhaohy


from .fm_function import fm_function
from .metrics import gini_norm
import tensorflow as tf

def main():

    key = 'FM,DeepFM,DNN'

    params = {
        "embedding_size": 8,
        "dropout_fm": [1.0, 1.0],
        "deep_layers": [32, 32],
        "dropout_deep": [0.8, 0.8, 0.8],
        "deep_layers_activation": tf.nn.relu,
        "epoch": 30,
        "batch_size": 1024,
        "learning_rate": 0.01,
        "optimizer_type": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        "l2_reg": 0.01,
        "verbose": True,
        "eval_metric": gini_norm,
        "random_seed": 33
    }


    fm_function(key,params)

main()