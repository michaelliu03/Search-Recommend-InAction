#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:train.py
# @Author: Michael.liu
# @Date:2020/6/15 17:13
# @Desc: this code is ....
import json
import time
import os
import argparse
import random
import tensorflow as tf
from .dssm_model import *
from .helper import *

class Trainer(object):
    def __init__(self,args):
        self.args = args
        with open(args.config_path,"r",encoding="utf8") as fr:
            self.config = json.load(fr)

        self.summaries_dir = self.config["summaries_dir"]



    def create_model(self):
        model = ""
        return model

    def train(self):

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9,allow_growth=True)
        sess_config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True,gpu_options=gpu_options)
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(self.summaries_dir+'/train',sess.graph)

            start = time.time()
            current_step = 0

            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))



