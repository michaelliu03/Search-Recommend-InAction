#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 
# @desc :
import os

import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .data_load import get_batch
from .model import Transformer
from .hparams import Hparams
from .utils import get_hypotheses,load_hparams
import logging

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser()
hp = parser.parse_args()
load_hparams(hp,hp.ckpt)

logging.info("# Prepare test batches")
test_batches,num_test_batches,num_test_samples = get_batch(hp.test1,hp.test1,
                                                           hp.maxlen1,hp.maxlen2,
                                                           hp.vocab,hp.test_batch_size,
                                                           shuffle=False)

iter = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.getoutput_types(test_batches),tf.compat.v1.data.get_output_shapes(test_batches))

test_init_op = iter.make_initializer(test_batches)

xs,ys = iter.get_next()

logging.info("# Load model")
m = Transformer(hp)
y_hat, _ = m.eval(xs,ys)

logging.info("# Session")
with tf.compat.v1.Session() as sess:
    ckpt_ = tf.compat.v1.train.latest_checkpoint(hp.ckpt)
    ckpt = hp.ckpt if ckpt_ is None else ckpt_
    saver = tf.compat.v1.train.Saver()

    saver.restore(sess,ckpt)
    model_bp = hp.ckpt_pb
    if not os.path.exists(model_bp):os.makedirs(model_bp)
    tf.io.write_graph(sess.graph_def,model_bp,'expert-graph.pb',as_text = False)

    sess.run(test_init_op)

    logging.info("# get hypotheses")
    hypotheses = get_hypotheses(num_test_batches,num_test_samples,sess,y_hat,m.idx2token)

    logging.info(type(hypotheses))
    logging.info("# write results")
    model_output = ckpt.split("/")[-1]
    if not os.path.exists(hp.testdir): os.makedirs(hp.testdir)
    translation = os.path.join(hp.testdir, model_output)
    with open(translation, 'w') as fout:
        fout.write("\n".join(hypotheses))

    logging.info("# calc bleu score and append it to translation")
    #calc_bleu(hp.test2, translation)





