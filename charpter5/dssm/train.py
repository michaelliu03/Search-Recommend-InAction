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
        model = DssmModel(config=self.config)
        return model

    def load_data(self):
        data_train = DssmData.load_data(self.config["train_file"])
        return data_train

    def train_steps(self):
        train_epoch_steps = int(len(self.data_train['query']) / query_BS) - 1
        return train_epoch_steps


    def train(self):
        self.data_train = self.load_data()
        self.data_epoch_steps = self.train_steps()
        self.model = self.create_model()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9,allow_growth=True)
        sess_config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True,gpu_options=gpu_options)
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(self.summaries_dir+'/train',sess.graph)

            start = time.time()
            current_step = 0

            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))
                batch_ids = [i for i in range(self.train_epoch_steps)]
                random.shuffle(batch_ids)
                for batch_id in batch_ids:
                    self.model.train(sess=sess,data_train=self.data_train,batch_id=self.batch_id)
                end = time.time()
                epoch_loss = 0
                for i in range(self.train_epoch_steps):
                    loss_v = sess.run(self.loss, feed_dict=self.feed_dict(False, data_train, i, 1))
                    epoch_loss += loss_v

                epoch_loss /= (self.train_epoch_steps)
                train_loss = sess.run(self.train_loss_summary, feed_dict={self.train_average_loss: epoch_loss})
                train_writer.add_summary(train_loss, epoch + 1)
                print("\nEpoch #%d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs" %
                      (epoch, epoch_loss, end - start))


if __name__ == "__main__":
    # 读取用户在命令行输入的信息
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="config path of model")
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()

