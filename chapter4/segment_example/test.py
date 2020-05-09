import time
import helper
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from .BILSTM_CRF import BILSTM_CRF
from .helper import Helper

# python test.py model test.in test.out -c char_emb -g 2

parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="the path of model file")
parser.add_argument("test_path", help="the path of test file")
parser.add_argument("output_path", help="the path of output file")
parser.add_argument("-c","--char_emb", help="the char embedding file", default=None)
parser.add_argument("-g","--gpu", help="the id of gpu, the default is 0", default=0, type=int)
args = parser.parse_args()

model_path = args.model_path
test_path = args.test_path
output_path = args.output_path
gpu_config = "/gpu:"+str(args.gpu)
emb_path = args.char_emb
num_steps = 200 # it must consist with the train

start_time = time.time()

helper = Helper()
print ("preparing test data")
helper.char2id, helper.id2char = helper.loadMap("char2id")
helper.label2id, helper.id2label = helper.loadMap("label2id")
x_test, y_test = helper.initFile(inputPath=test_path, seqMaxLen=num_steps)
num_chars = len(helper.id2char.keys())
num_classes = len(helper.id2label.keys())
if emb_path != None:
	#embedding_matrix = helper.getEmbedding(emb_path)
	embedding_matrix = None
else:
	embedding_matrix = None

print("building model")
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
	with tf.device(gpu_config):
		initializer = tf.random_uniform_initializer(-0.1, 0.1)
		with tf.variable_scope("model", reuse=None, initializer=initializer):
			model = BILSTM_CRF(num_chars=num_chars, num_classes=num_classes, helper=helper, num_steps=num_steps, embedding_matrix=embedding_matrix, is_training=False)

		print ("loading model parameter")
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		print ("testing")
		model.test(sess, output_path)

		end_time = time.time()
		print ("time used %f(hour)" % ((end_time - start_time) / 3600))
