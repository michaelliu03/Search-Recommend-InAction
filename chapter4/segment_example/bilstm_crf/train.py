import time
import helper
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from .BILSTM_CRF import BILSTM_CRF
from .helper import Helper

# python lr_train.py train.in model -v validation.in -c char_emb -e 10 -g 2

parser = argparse.ArgumentParser()
parser.add_argument("train_path", help="the path of the train file")
parser.add_argument("save_path", help="the path of the saved model")
parser.add_argument("-v","--val_path", help="the path of the validation file", default=None)
parser.add_argument("-e","--epoch", help="the number of epoch", default=100, type=int)
parser.add_argument("-c","--char_emb", help="the char embedding file", default=None)
parser.add_argument("-g","--gpu", help="the id of gpu, the default is 0", default=0, type=int)

args = parser.parse_args()

train_path = args.train_path
save_path = args.save_path
val_path = args.val_path
num_epochs = args.epoch
emb_path = args.char_emb
gpu_config = "/gpu:"+str(args.gpu)
num_steps = 200 # it must consist with the test

start_time = time.time()

helper = Helper()
print ("preparing train and validation data")
helper.buildMap(trainPath=train_path)
x_train, y_train, x_val, y_val = helper.initFile(inputPath=train_path, validPath=val_path, seqMaxLen=num_steps)
num_chars = len(helper.id2char.keys())
num_classes = len(helper.id2label.keys())
print ('num_chars size = %d, num_classes=%d' % (num_chars, num_classes))
if emb_path != None:
    #embedding_matrix = helper.getEmbedding(emb_path)
    embedding_matrix = None
else:
    embedding_matrix = None

print ("building model")
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    with tf.device(gpu_config):
        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            model = BILSTM_CRF(num_chars=num_chars, num_classes=num_classes, helper=helper, num_steps=num_steps, num_epochs=num_epochs, embedding_matrix=embedding_matrix, is_training=True)

            print ("training model")
            tf.initialize_all_variables().run()
            model.train(sess, save_path, x_train, y_train, x_val, y_val)

            print ("final best f1 is: %f" % (model.max_f1))

            end_time = time.time()
            print ("time used %f(hour)" % ((end_time - start_time) / 3600))
