#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:helper.py
# @Author: Michael.liu
# @Date:2020/6/11 15:09
# @Desc: this code is ....
import re
from pyhanlp import *
import random
from tensorflow.contrib import learn
import numpy as np
import os
import json
import copy
import random
from collections import Counter
from itertools import chain


from gensim import corpora, models


class Data_Helper(object):

    def __init__(self,config):
        HanLP = SafeJClass('com.hankcs.hanlp.HanLP')

        self.__output_path = config["output_path"]
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)

        self.__neg_samples = config["neg_samples"]
        self.__n_tasks = config["n_tasks"]
        self.__stop_word_path = config["stop_word_path"]
        self.__low_freq = config["low_freq"]

        self.vocab_size = None
        self.word_vectors = None
        self.count = 0

        self.dictionary, self.tf_idf_model = None, None

    '''
           训练模型
           保存模型
       '''

    # def pre_processing(self):
    #     train_texta, train_textb, train_tag = self.load_data('train.txt')
    #     data = []
    #     data.extend(train_texta)
    #     data.extend(train_textb)
    #     self.build_vocab(data, '/vocab.pickle')
    #     # 加载词典
    #     self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore('/vocab.pickle')
    #     train_texta_embedding = np.array(list(self.vocab_processor.transform(train_texta)))
    #     train_textb_embedding = np.array(list(self.vocab_processor.transform(train_textb)))
    #
    #     dev_texta, dev_textb, dev_tag = self.load_data('dev.txt')
    #     dev_texta_embedding = np.array(list(self.vocab_processor.transform(dev_texta)))
    #     dev_textb_embedding = np.array(list(self.vocab_processor.transform(dev_textb)))
    #     return train_texta_embedding, train_textb_embedding, np.array(train_tag), \
    #            dev_texta_embedding, dev_textb_embedding, np.array(dev_tag)

    @staticmethod
    def load_data(file_path):
        """

        :param file_path:
        :return:
        """
        queries = []

        return queries

    def neg_samples(self, queries):
        """
        随机负采样多个样本
        :param queries:
        :return:
        """
        new_queries = []
        new_sims = []
        for_nums = self.__n_tasks // len(queries)

        for i in range(for_nums):
            for questions in queries:
                copy_questions = copy.copy(queries)
                copy_questions.remove(questions)
                pos_samples = random.sample(questions, 2)

                copy_questions = list(chain(*copy_questions))
                neg_sims = random.sample(copy_questions, self.__neg_samples)
                new_queries.append(pos_samples[0])
                new_sims.append([pos_samples[1]] + neg_sims)
        return new_queries, new_sims

    def remove_stop_word(self, inputs):
        """
        去除低频词/字和停用词/字, 构建一个项目的vocab词/字汇表
        :param inputs: 输入
        :return:
        """
        all_words = list(chain(*inputs))

        word_count = Counter(all_words)  # 统计词频
        sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        words = [item[0] for item in sort_word_count if item[1] > self.__low_freq]

        # 如果传入了停用词表，则去除停用词
        if self.__stop_word_path:
            with open(self.__stop_word_path, "r", encoding="utf8") as fr:
                stop_words = [line.strip() for line in fr.readlines()]
            words = [word for word in words if word not in stop_words]

        return words

    @staticmethod
    def trans_to_tf_idf(inputs, dictionary, tf_idf_model):
        vocab_size = len(dictionary.token2id)
        input_ids = []
        for questions in inputs:
            question_ids = []
            for question in questions:
                bow_vec = dictionary.doc2bow(question)
                tfidf_vec = tf_idf_model[bow_vec]
                vec = [0] * vocab_size
                for item in tfidf_vec:
                    vec[item[0]] = item[1]
                question_ids.append(vec)
            input_ids.append(question_ids)
        return input_ids

    @staticmethod
    def train_tf_idf(inputs):
        sentences = list(chain(*inputs))
        dictionary = corpora.Dictionary(sentences)
        corpus = [dictionary.doc2bow(sentence) for sentence in sentences]
        tfidf_model = models.TfidfModel(corpus)
        return dictionary, tfidf_model

    def gen_data(self, file_path):
        """

        :param file_path:
        :return:
        """
        inputs = self.load_data(file_path)
        self.dictionary, self.tf_idf_model = self.train_tf_idf(inputs)
        input_ids = self.trans_to_tf_idf(inputs, self.dictionary, self.tf_idf_model)
        return input_ids

    def next_batch(self, query_ids, batch_size):
        """

        :param query_ids:
        :param batch_size:
        :return:
        """
        x, y = self.neg_samples(query_ids)
        self.count += 1
        print("sample: {}".format(self.count))
        num_batches = len(x) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = x[start: end]
            batch_y = y[start: end]
            new_batch_y = []
            for item in batch_y:
                new_batch_y += item
            yield dict(query=batch_x, sim=new_batch_y)



    def load_data(self, fileName):
        text_a = []
        text_b = []
        tag = []
        with open(fileName,'r') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                #print(line)
                text_a.append(self.preprocess(line[0]))
                text_b.append(self.preprocess(line[1]))
                tag.append(line[2])
        # shuffle
        index = [x for x in range(len(text_a))]
        random.shuffle(index)
        text_a_new = [text_a[x] for x in index]
        text_b_new = [text_b[x] for x in index]
        tag_new =[tag[x] for x in index]

        type = list(set(tag_new))
        dicts = {}
        tags_vec = []
        for  x in tag_new:
            if x not in dicts.keys():
                dicts[x] = 1
            else:
                dicts[x] +=1
            temp = [0] * len(type)
            temp[int(x)] =1
            tags_vec.append(temp)
        print(dicts)
        return text_a_new,text_b_new,tags_vec

    def seg_formate(self,content):
        ret = []
        seg_list = HanLP.segment(content)
        for term in seg_list:
            word = str(term.word).strip().lower()
            ret.append(word)
        seg_content = ' '.join(ret)
        return seg_content

    # text preprocess
    def preprocess(self,text):
        HanLP = SafeJClass('com.hankcs.hanlp.HanLP')
        ret = []
        text = re.sub('（[^（.]*）','',text)
        # 只保留中文部分
        text = ''.join([x for x in text if '\u4e00' <= x <= '\u9fa5'])
        # 分词
        words = self.seg_formate(text)
        # 不分词
        words = [x for x in ''.join(words)]
        return ' '.join(words)

    def build_vocab(self, sentences, path):
        lens = [len(sentence.split(" ")) for sentence in sentences]
        max_length = max(lens)
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)
        vocab_processor.fit(sentences)
        vocab_processor.save(path)

if __name__ == '__main__':
    print(".....start.....")
    data_helper = Data_Helper()
    data_helper.load_data('xxx')
    print("end......")
