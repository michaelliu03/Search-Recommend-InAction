#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 
# @desc :

import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .model import Transformer
from tqdm import tqdm
from .data_load import get_batch
from .utils import save_hparams,save_variable_specs,get_hypotheses
import os
import sys
from .hparams import Hparams
import math
import logging
import gc
from tensorflow.compat.v1 import keras

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp,hp.ckpt)
tf.reset_default_graph()
logging.info('# Prepare train / eval batches')
train_batches,num_train_batches,num_train_samples = get_batch(hp.train1,hp.train2,
                                                              hp.maxlen1,hp.maxlen2,
                                                              hp.vocab,hp.batch_size,shuffle=True)

eval_batches,num_eval_batches,num_eval_samples = get_batch(hp.eval1,hp.eval2,
                                                           hp.maxlen1,hp.maxlen2,
                                                           hp.vocab,hp.batch_size,
                                                           shuffle= False)
print("eval: ",num_eval_batches,num_eval_samples)

iter = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(train_batches),tf.compat.v1.data.get_output_shapes(train_batches))

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

xs,ys = iter.get_next()

logging.info("# Load model")

m = Transformer(hp)

loss, train_op, global_step,train_summaries = m.train(xs,ys)
print("loss: ",loss)
y_hat,eval_summaries = m.eval(xs,ys)
print("eval y_hay: ",y_hat)

logging.info("# Session")
saver = tf.compat.v1.train.Saver(max_to_keep= hp.num_epochs)
with tf.compat.v1.Session() as sess:
    ckpt = tf.compat.v1.train.latest_checkpoint(hp.ckpt)
    if ckpt is None:
        logging.info("Initalizing from scratch")
        sess.run(tf.compat.v1.global_variables_initializer())
        save_variable_specs(os.path.join(hp.ckpt, "specs"))
    else:
        saver.restore(sess, ckpt)

    summary_writer = tf.compat.v1.summary.FileWriter(hp.ckpt, sess.graph)

    sess.run(train_init_op)
    total_steps = hp.num_epochs  * num_train_batches
    _gs = sess.run(global_step)
    print(_gs)

    for i in tqdm(range(_gs,total_steps +1)):
        _, _gs, _summary = sess.run([train_op,global_step,train_summaries])
        epoch = math.ceil(_gs / num_train_batches)
        summary_writer.add_summary(_summary,_gs)

        if _gs and _gs % num_train_batches == 0:
            logging.info("epoch {} is done".format(epoch))
            _loss = sess.run(loss)

            logging.info("# test evaluation")
            _,_eval_summaries = sess.run([eval_init_op,eval_summaries])
            summary_writer.add_summary(_eval_summaries, _gs)

            logging.info("# get hypotheses")
            hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, m.idx2token)

            logging.info("# write results")
            model_output = "iwslt2016_E%02dL%.2f" % (epoch, _loss)
            if not os.path.exists(hp.evaldir): os.makedirs(hp.evaldir)
            translation = os.path.join(hp.evaldir, model_output)
            with open(translation, 'w') as fout:
                fout.write("\n".join(hypotheses))

            logging.info("# calc bleu score and append it to translation")

            logging.info("# save models")
            ckpt_name = os.path.join(hp.ckpt, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            logging.info("# fall back to train mode")
            sess.run(train_init_op)

    model_bp = hp.ckpt_pb
    if not os.path.exists(model_bp): os.makedirs(model_bp)
    tf.io.write_graph(sess.graph_def, model_bp, 'expert-graph.pb', as_text=False)
    # tf.saved_model.save(m, model_bp)
    summary_writer.close()

logging.info("Done !")

sys.exit()

logging.info("Done2 !")

a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

sess = tf.compat.v1.Session()

print(sess.run(a))
print(sess.run(c))


