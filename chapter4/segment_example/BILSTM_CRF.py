import math
import helper
import numpy as np
import tensorflow as tf
#from tensorflow.models.rnn import rnn, rnn_cell

class BILSTM_CRF(object):
    
    def __init__(self, num_chars, num_classes, helper, num_steps=200, num_epochs=100, embedding_matrix=None, is_training=True, is_crf=True, weight=False):
        # Parameter
        self.max_f1 = 0
        self.learning_rate = 0.002
        self.dropout_rate = 0.5
        self.batch_size = 128

        self.num_layers = 1   
        self.emb_dim = 100
        self.hidden_dim = 100
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.num_chars = num_chars
        self.num_classes = num_classes
        self.helper = helper
        
        # placeholder of x, y and weight
        self.inputs = tf.placeholder(tf.int32, [None, self.num_steps])
        self.targets = tf.placeholder(tf.int32, [None, self.num_steps])
        
        # char embedding
        if embedding_matrix != None:
            self.embedding = tf.Variable(embedding_matrix, trainable=False, name="emb", dtype=tf.float32)
        else:
            self.embedding = tf.get_variable("emb", [self.num_chars, self.emb_dim])
        self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)
        self.inputs_emb = tf.transpose(self.inputs_emb, [1, 0, 2])
        self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.emb_dim])
        self.inputs_emb = tf.split(self.inputs_emb, self.num_steps, 0)

        # lstm cell
        #lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)
        #lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)
        lstm_cell_fw = tf.contrib.rnn.GRUCell(self.hidden_dim)
        lstm_cell_bw = tf.contrib.rnn.GRUCell(self.hidden_dim)

        # dropout
        if is_training:
            lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout_rate))
            lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout_rate))

        lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell_fw] * self.num_layers)
        lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell_bw] * self.num_layers)

        # get the length of each sample
        self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
        self.length = tf.cast(self.length, tf.int32)  
        
        # forward and backward
        self.outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
            lstm_cell_fw, 
            lstm_cell_bw,
            self.inputs_emb, 
            dtype=tf.float32,
            sequence_length=self.length
        )
        
        # softmax
        self.outputs = tf.reshape(tf.concat(self.outputs, 1), [-1, self.hidden_dim * 2])
        self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim * 2, self.num_classes])
        self.softmax_b = tf.get_variable("softmax_b", [self.num_classes])
        self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b

        self.tags_scores = tf.reshape(self.logits, [self.batch_size, self.num_steps, self.num_classes])
        # loss            
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.tags_scores, self.targets, self.length)
        self.loss = tf.reduce_mean(-log_likelihood)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss) 


    def train(self, sess, save_file, X_train, y_train, X_val, y_val):
        saver = tf.train.Saver()

        print ("load char2id")
        char2id, id2char = self.helper.loadMap("char2id")
        label2id, id2label = self.helper.loadMap("label2id")

        num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size))

        print ("start training...")
        cnt = 0
        for epoch in range(self.num_epochs):
            # shuffle train in each epoch
            #sh_index = np.arange(len(X_train))
            #np.random.shuffle(sh_index)
            #X_train = X_train[sh_index]
            #y_train = y_train[sh_index]
            self.helper.shuffle()
            print ("current epoch: %d" % (epoch))
            for iteration in range(num_iterations):
                # train
                X_train_batch, y_train_batch = self.helper.nextBatch(batch_size=self.batch_size)
                
                _, current_transition_params, length =\
                    sess.run([
                        self.optimizer, 
                        self.transition_params,  
                        self.length
                    ], 
                    feed_dict={
                        self.inputs:X_train_batch, 
                        self.targets:y_train_batch, 
                    })

                current_tags_scores, predict_lengths = sess.run([self.tags_scores, self.length], feed_dict={self.inputs:X_train_batch,})
                predict_sequences = []
                x_train_batch_cut = []
                y_train_batch_cut = []
                for un_current_tags_scores, predict_length, x, y in zip(current_tags_scores, predict_lengths, X_train_batch, y_train_batch):
                	   predict_sequence, _ = tf.contrib.crf.viterbi_decode(un_current_tags_scores[:predict_length], current_transition_params)
                	   predict_sequences.append(predict_sequence)
                	   x_train_batch_cut.append(x[:predict_length])
                	   y_train_batch_cut.append(y[:predict_length])
                
                if iteration % 10 == 0:
                    cnt += 1
                    precision_train, recall_train, f1_train = self.evaluate(x_train_batch_cut, y_train_batch_cut, predict_sequences, id2char, id2label)
                    #print "iteration: %5d, train loss: %5d, train precision: %.5f, train recall: %.5f, train f1: %.5f" % (iteration, loss_train, precision_train, recall_train, f1_train)
                    print ("iteration: %5d, train loss: null, train precision: %.5f, train recall: %.5f, train f1: %.5f" % (iteration, precision_train, recall_train, f1_train)  )
                    
                # validation
                if iteration % 100 == 0:
                    X_val_batch, y_val_batch = self.helper.nextRandomBatch(batch_size=self.batch_size)
                    
                    current_tags_scores, predict_lengths = sess.run([self.tags_scores, self.length], feed_dict={self.inputs:X_val_batch,})
                    predict_sequences = []
                    x_val_batch_cut = []
                    y_val_batch_cut = []
                    for un_current_tags_scores, predict_length, x, y in zip(current_tags_scores, predict_lengths, X_val_batch, y_val_batch):
                	      predict_sequence, _ = tf.contrib.crf.viterbi_decode(un_current_tags_scores[:predict_length], current_transition_params)
                	      predict_sequences.append(predict_sequence)
                	      x_val_batch_cut.append(x[:predict_length])
                	      y_val_batch_cut.append(y[:predict_length])
                    precision_val, recall_val, f1_val = self.evaluate(x_val_batch_cut, y_val_batch_cut, predict_sequences, id2char, id2label)
                    print ("iteration: %5d, valid precision: %.5f, valid recall: %.5f, valid f1: %.5f" % (iteration, precision_val, recall_val, f1_val))

                    if f1_val > self.max_f1:
                        self.max_f1 = f1_val
                        save_path = saver.save(sess, save_file)
                        print("path = %s" % (save_path))
                        print ("saved the best model with f1: %.5f" % (self.max_f1))

    def test(self, sess, output_path):
        char2id, id2char = self.helper.loadMap("char2id")
        label2id, id2label = self.helper.loadMap("label2id")
        num_iterations = int(math.ceil(1.0 * len(self.helper.inputX) / self.batch_size))
        print ("number of iteration: " + str(num_iterations))
        with open(output_path, "w") as outfile:
            for i in range(num_iterations):
                print ("iteration: " + str(i + 1))
                lastIndex = self.helper.inputIndex
                x_test_batch, y_test_batch = self.helper.nextBatch(batch_size=self.batch_size)
                results = self.predictBatch(sess, x_test_batch, id2char, id2label)

                if i == num_iterations - 1:
                    last_size = len(self.helper.inputX) % self.batch_size
                    if last_size > 0: 
                        results = results[:last_size]
                
                for i in range(len(results)):
                    outfile.write(" ".join(results[i]) + "\n")


    def predictBatch(self, sess, x_test_batch, id2char, id2label):
        results = []

        current_tags_scores, predict_lengths, current_transition_params = sess.run([self.tags_scores, self.length, self.transition_params], feed_dict={self.inputs:x_test_batch,})
        predicts = []
        for un_current_tags_scores, predict_length in zip(current_tags_scores, predict_lengths):
            if predict_length == 0: continue
            predict_sequence, _ = tf.contrib.crf.viterbi_decode(un_current_tags_scores[:predict_length], current_transition_params)
            predicts.append(predict_sequence)

        for i in range(len(predicts)):
            predict_length = len(predicts[i])
            if predict_length == 0: continue
            words = self.tag2words(x_test_batch[i][:predict_length], predicts[i], id2char, id2label)
            results.append(words.split())
        return results

    def tag2words(self, xs, ys, id2char, id2label):
        words = ''
        for x, y in zip(xs, ys):
            ch = str(id2char[x])
            label = str(id2label[y])
            if label == 'B': words += ' ' + ch
            elif label == 'M': words += ch
            elif label == 'E': words += ch + ' '
            elif label == 'S': words += ' ' + ch + ' '
        return words

    def evaluate(self, X, y_true, y_pred, id2char, id2label):
        precision = -1.0
        recall = -1.0
        f1 = -1.0
        hit_num = 0
        pred_num = 0
        true_num = 0
        pred_words = ''
        true_words = ''

        for i in range(len(X)):
            xs = X[i]
            #print xs
            true_words = self.tag2words(xs, y_true[i], id2char, id2label)
            pred_words = self.tag2words(xs, y_pred[i], id2char, id2label)

            trueWordSet = set()
            i = 0
            true_num += len(true_words.split())
            for word in true_words.split():
                trueWordSet.add(word + '-' + str(i))
                i += len(word)

            i = 0
            pred_num += len(pred_words.split())
            for word in pred_words.split():
                term = word + '-' + str(i)
                i += len(word)
                if term in trueWordSet: hit_num+= 1

        if pred_num != 0:
            precision = 1.0 * hit_num / pred_num
        if true_num != 0:
            recall = 1.0 * hit_num / true_num
        if precision > 0 and recall > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        return precision, recall, f1
