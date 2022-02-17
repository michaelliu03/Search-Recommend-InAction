#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 
# @desc :
from transformers import XLNetTokenizer, TFXLNetModel
import os
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from queue import Queue
from threading import Thread
import json



class Xlnet:
    def __int__(self,index = None):
        if index is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(index)
        config = tf.compat.v1.ConfigProto(log_device_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        config.gpu_options.allocator_type = 'BFC'  # 将内存分块管理，按块进行空间分配和释放
        # config.allow_soft_placement=True
        set_session(tf.compat.v1.Session(config=config))

        self.model = TFXLNetModel.from_pretrained('xlnet-large-cased')
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased', do_lower_case=True)
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)
        self.predict_thread = Thread(target=self.predict_from_queue, daemon=True)
        self.predict_thread.start()

    def encode(self, sentences):
        self.input_queue.put(sentences)
        output = self.output_queue.get()
        return output

    def predict_from_queue(self):
        while True:
            sentences = self.input_queue.get()
            encoded_input = self.tokenizer(sentences, return_tensors='tf', padding=True)
            outputs = self.model(encoded_input)
            last_hidden_states = outputs.last_hidden_state
            pooled_sentence = [tf.reduce_mean(vector, 0).numpy().tolist() for vector in last_hidden_states]
            self.output_queue.put(pooled_sentence)





