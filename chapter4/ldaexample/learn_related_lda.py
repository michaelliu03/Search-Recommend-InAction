#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:learn_related_lda.py
# @Author: Michael.liu
# @Date:2020/4/26 16:59
# @Desc: this is the lda_example

import codecs
from gensim import corpora,models,similarities
from pprint import pprint
import time
import numpy as np
import configparser


text_num =[] # 文本的个数
dic_vol = [] # 词典的容量

stop_words = set()
def read_stop_words(filepath):
    f = open(filepath, 'r',encoding='utf-8')
    words = f.read()
    stop_words = set(words.split('\n'))


class LDAModel(object):
    # 定义文章的个数
    _M = []
    _V = []
    filePath = ""
    document = []
    _texts =[]
    _dictionary=[]
    _corpus=[]
    _lda=[]
    _num_topics=0
    _corpus_tfidf=[]
    _num_show_term = 0
    config_path = ''
    config_encoding = ''
    _config =''


    def __init__(self,corpusfilepath,config_path,config_encoding):
        t_start = time.time()
        self.config_path = config_path
        self.config_encoding = config_encoding
        self._config = configparser.ConfigParser()
        self._config.read(config_path, config_encoding)
        #print(config_path + "\t" + config_encoding)

        self.filePath = corpusfilepath
        #转变文章的格式
        data = codecs.open(filePath, 'r', encoding="utf-8")
        for item in data:
            self.document.append(item)

        # 语料已经做过一次停用词过滤
        self._texts = [[word for word in line.strip().lower().split() if word not in stop_words] for line in self.document]

        data.close()
        print('1.读入语料数据完成，用时%.3f秒' % (time.time() - t_start))

    def build_vector(self):
        t_start = time.time()
        self._M = len(self._texts)
        print('文本的个数: %d 个' % self._M)
        self._dictionary = corpora.Dictionary(self._texts)
        V = len(self._dictionary)
        print('词典维度为: %d 个' % V)
        print('建立文本向量')
        self._corpus = [self._dictionary.doc2bow(text) for text in self._texts]
        #print(self._corpus)
        print('完成向量构建,用时%.3f秒' % (time.time() - t_start))

    def train(self,num_topics):
        t_start = time.time()
        # 计算tf-idf值
        self._corpus_tfidf = models.TfidfModel(self._corpus)[self._corpus]
        print('建立文档TF-IDF完成，用时%.3f秒' % (time.time() - t_start))

        print('6.LDA模型拟合推断 ------')
        # 训练模型
        self._num_topics = num_topics
        print(self.Config)

        self._lda = models.LdaModel(self._corpus_tfidf,
                              num_topics=self._num_topics,
                              id2word=self._dictionary,
                              alpha=0.01,
                              eta=0.01,
                              minimum_probability=0.001,
                              update_every=1,
                              chunksize=100,
                              passes=1)
        print('LDA模型完成，训练时间为\t%.3f秒' % (time.time() - t_start))
        return self._lda


    @property
    def M(self):
        return self._M

    @property
    def V(self):
        return self._V

    @property
    def Config(self):
        return self._config

    def show_topic(self,num_show_topic):
        t_start = time.time()
        print('7.结果：10个文档的主题分布：--')
        doc_topics = self._lda.get_document_topics(self._corpus_tfidf)  # 所有文档的主题分布
        idx = np.arange(self._M)
        np.random.shuffle(idx)
        idx = idx[:10]
        for i in idx:
            topic = np.array(doc_topics[i])
            topic_distribute = np.array(topic[:, 1])
            # print topic_distribute
            topic_idx = topic_distribute.argsort()[:-num_show_topic - 1:-1]
            print('第%d个文档的前%d个主题：' % (i, num_show_topic)), topic_idx
            print(topic_distribute[topic_idx])
        print('show_topic，\t%.3f' % (time.time() - t_start))

    def show_topic_words(self,num_show_term):
        t_start = time.time()
        self._num_show_term = num_show_term  # 每个主题显示几个词
        print('结果：每个主题的词分布：--')
        for topic_id in range(self._num_topics):
            print('主题#%d：\t' % topic_id)
            term_distribute_all = self._lda.get_topic_terms(topicid=topic_id)
            term_distribute = term_distribute_all[:num_show_term]
            term_distribute = np.array(term_distribute)
            term_id = term_distribute[:, 0].astype(np.int)
            print('词：\t', )
            for t in term_id:
                print(self._dictionary.id2token[t], )
            print('\n概率：\t', term_distribute[:, 1])
        print('show_topic_words，\t%.3f' % (time.time() - t_start))


if __name__ == "__main__":
    print(".....this is the begin.....")
    # step 1:
    stop_filepath = u'../../chapter2/stop_words.utf8'
    filePath = u'./lda_prepare.csv'
    read_stop_words(stop_filepath)
    ldamodel = LDAModel(filePath,'./config.ini','utf-8')
    ldamodel.build_vector()
    ldamodel.train(30)
    ldamodel.show_topic(10)
    ldamodel.show_topic_words(10)

    #build_tfidf(filePath,100,10)
    print("finished!!")