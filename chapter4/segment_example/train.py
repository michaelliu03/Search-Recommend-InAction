#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:train.py
# @Author: Michael.liu
# @Date:2020/5/8 17:04
# @Desc: this code is ....

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from .bilstm import BiLSTM_CRF
import pickle
import os
ops.reset_default_graph()
data_folder_name = '..\\temp'
data_path_name = 'segmentation'

tfrecord_train_name = 'people_daily_train.tfrecord'
tfrecord_test_name = 'people_daily_test.tfrecord'
# ckpt_name = 'cbow_movie_embeddings.ckpt'

save_ckpt_name = 'bilstm_crf_cn.ckpt'
vocab_name = 'bilstm_crf_cn.pkl'
model_log_name = 'model_log.txt'
tensorboard_name = 'tensorboard'

tensorboard_path = os.path.join(data_folder_name, data_path_name, tensorboard_name)
model_log_path = os.path.join(data_folder_name, data_path_name, model_log_name)
vocab_path = os.path.join(data_folder_name, data_path_name, vocab_name)
with open(vocab_path, 'rb') as f:
    word_dict = pickle.load(f)
batch_size = 512
test_batch_size = 2048
lstm_dim = 100
embedding_size = 100
vocabulary_size = len(word_dict)

generations = 1800
num_tags = 4
shuffle_pool_size = 5000
max_len = 1039
save_flag = True

sess = tf.Session()
summary_writer = tf.summary.FileWriter(tensorboard_path, tf.get_default_graph())


def __parse_function(serial_exmp):
    features = tf.parse_single_example(serial_exmp, features={"text": tf.VarLenFeature(tf.int64),
                                                              "label": tf.VarLenFeature(tf.int64),
                                                              "length": tf.FixedLenFeature([], tf.int64)})
    # text = tf.sparse_tensor_to_dense(features["text"], default_value=" ")
    text_ = tf.sparse.to_dense(features["text"])
    label_ = tf.sparse.to_dense(features["label"])
    lens_ = tf.cast(features["length"], tf.int32)
    return text_, label_, lens_


def get_dataset(tf_record_name):
    tf_record_path = os.path.join(data_folder_name,data_path_name, tf_record_name)
    dataset = tf.data.TFRecordDataset(tf_record_path)
    return dataset.map(__parse_function)


def viterbi_decode(score_, trans_):
    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(score_, trans_)
    return viterbi_sequence


def evaluate(scores_, lengths_, trans_, targets_):
    correct_seq = 0.
    total_len = 0.
    for ix, score_ in enumerate(scores_):
        score_real = score_[:lengths_[ix]]
        target_real = targets_[ix][:lengths_[ix]]
        pre_sequence = viterbi_decode(score_real, trans_)
        correct_seq += np.sum((np.equal(pre_sequence, target_real)))
        total_len += lengths_[ix]
    return correct_seq/total_len


train_data_set = get_dataset(tfrecord_train_name)
train_data = train_data_set.shuffle(shuffle_pool_size).\
    padded_batch(batch_size, padded_shapes=([max_len], [max_len], [])).repeat()
train_iter = train_data.make_one_shot_iterator()
train_handle = sess.run(train_iter.string_handle())

test_data_set = get_dataset(tfrecord_test_name)
test_data = test_data_set.\
    padded_batch(test_batch_size, padded_shapes=([max_len], [max_len], [])).repeat()
test_iter = test_data.make_one_shot_iterator()
test_handle = sess.run(test_iter.string_handle())

handle = tf.placeholder(tf.string, shape=[], name='handle')
iterator = tf.data.Iterator.from_string_handle(handle, train_data.output_types, train_data.output_shapes)
x_input, y_target, lens = iterator.get_next()

embeddings = tf.Variable(tf.truncated_normal(
    [vocabulary_size, embedding_size], stddev=0.05))

lstm_model = BiLSTM_CRF(embeddings)
logits = lstm_model.logits
loss = lstm_model.loss
train_step = lstm_model.train_step

with tf.name_scope('Loss_and_Accuracy'):
    tf.summary.scalar('Loss', loss)
    # tf.summary.scalar('Acc', loss)
summary_op = tf.summary.merge_all()

sess.run(tf.global_variables_initializer())

print('start training')
saver = tf.train.Saver(max_to_keep=1)
max_accuracy = 0
temp_train_loss = []
temp_test_loss = []
temp_train_acc = []
temp_test_acc = []
for i in range(generations):
    x_batch, y_batch, batch_len = sess.run([x_input, y_target, lens],
                                           feed_dict={handle: train_handle})
    feed_dict = {lstm_model.x_input: x_batch, lstm_model.y_target: y_batch,
                 lstm_model.dropout: 0.5, lstm_model.max_steps: batch_len}
    sess.run(train_step, feed_dict)
    train_feed_dict = {lstm_model.x_input: x_batch, lstm_model.y_target: y_batch,
                       lstm_model.dropout: 1.0, lstm_model.max_steps: batch_len}
    train_loss, train_logits, trans_martix = sess.run([loss, logits, lstm_model.trans], train_feed_dict)
    train_accuracy = evaluate(train_logits, batch_len, trans_martix, y_batch)
    print('Generation # {}. Train Loss : {:.3f} . '
          'Train Acc : {:.3f}'.format(i, train_loss, train_accuracy))
    temp_train_loss.append(train_loss)
    temp_train_acc.append(train_accuracy)
    summary_writer.add_summary(sess.run(summary_op, train_feed_dict), i)
    if (i+1) % 10 == 0:
        test_x_batch, test_y_batch, test_batch_len = sess.run([x_input, y_target, lens],
                                                              feed_dict={handle: test_handle})
        test_feed_dict = {lstm_model.x_input: test_x_batch, lstm_model.y_target: test_y_batch,
                          lstm_model.dropout: 1.0, lstm_model.max_steps: test_batch_len}

        test_loss, test_logits = sess.run([loss, logits], test_feed_dict)
        test_accuracy = evaluate(test_logits, test_batch_len, trans_martix, test_y_batch)
        print('Generation # {}. Test Loss : {:.3f} . '
              'Test Acc : {:.3f}'.format(i, test_loss, test_accuracy))
        temp_test_loss.append(test_loss)
        temp_test_acc.append(test_accuracy)
        if test_accuracy >= max_accuracy and save_flag:
            max_accuracy = test_accuracy
            saver.save(sess, os.path.join(data_folder_name, data_path_name, save_ckpt_name))
            print('Generation # {}. --model saved--'.format(i))
print('Last accuracy : ', max_accuracy)
with open(model_log_path, 'w') as f:
    f.write('train_loss: '+str(temp_train_loss))
    f.write('\n\ntest_loss: '+str(temp_test_loss))
    f.write('\n\ntrain_acc: '+str(temp_train_acc))
    f.write('\n\ntest_acc: '+str(temp_test_acc))
print(' --log saved--')

