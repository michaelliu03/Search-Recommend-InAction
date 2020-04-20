#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:learn_related.py
# @Author: Michael.liu
# @Date:2019/2/12
# @Desc:

from os import listdir
import xml.etree.ElementTree as ET
from pyhanlp import *
import sqlite3
import configparser
from datetime import *
import math
import pandas as pd
import numpy as np
from pyhanlp import *
from sklearn.metrics import pairwise_distances
import jieba.analyse


class LearnRanking:
    stop_words = set()
    k_nearest = []
    config_path =''
    config_encoding = ''
    doc_dir_path = ''
    doc_encoding = ''
    stop_words_path = ''
    stop_words_encoding= ''
    idf_path= ''
    db_path = ''
    big_dic={}

    def __init__(self,config_path,config_encoding):
        self.config_path = config_path
        self.config_encoding = config_encoding
        config = configparser.ConfigParser()
        config.read(config_path,config_encoding)
        # self.doc_dir_path = config['DEFAULT']['doc_dir_path']
        # self.doc_encoding = config['DEFAULT']['doc_encoding']

        file_path = os.path.join(os.path.dirname(__file__),config['DEFAULT']['stop_words_path'])
        file_encoding =config['DEFAULT']['stop_words_encoding']
        f = open(file_path, encoding=file_encoding)
        words = f.read()
        self.stop_words = set(words.split('\n'))
        self.db_path = config['DEFAULT']['db_path']

        config = configparser.ConfigParser()
        config.read(self.config_path, self.config_encoding)
        self.doc_dir_path = config['DEFAULT']['doc_dir_path']
        self.idf_path = config['DEFAULT']['idf_path']




    def write_k_nearest_matrix_to_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''DROP TABLE IF EXISTS knearest''')
        c.execute('''CREATE TABLE knearest
                     (id INTEGER PRIMARY KEY, first INTEGER, second INTEGER,
                     third INTEGER, fourth INTEGER, fifth INTEGER)''')

        for docid, doclist in self.k_nearest:
            c.execute("INSERT INTO knearest VALUES (?, ?, ?, ?, ?, ?)", tuple([docid] + doclist))

        conn.commit()
        conn.close()

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def extract_keywords(self,content,N):
         ret = [] #最终结果
         HanLP.Config.ShowTermNature = True  # 关闭词性
         TextRankKeyword = JClass("com.hankcs.hanlp.summary.TextRankKeyword")
         #这里的逻辑不能直接使用pyhanlp的抽取关键词的逻辑，
         #第一步预处理 #第二部计算tfidf
         text = content.lower()
         word_li = []
         keyword_list = HanLP.extractKeyword(content, N)
         ret = keyword_list
         return ret

    def new_seg(self, content):
        ret_list = []
        seg_list = HanLP.segment(content)
        for term in seg_list:
            word = str(term.word)
            # 判断新
            if word == '' or word == '\r' or word == '\t\n' or word == '\n' or word == '\t':
                continue
            elif len(word) == 1:
                continue
            else:
                ret_list.append(word)
        #ret = ' '.join(ret_list)
        return ret_list



    def clean_list(self, seg_list):
        cleaned_dict = {}
        n = 0
        for i in seg_list:
            i = i.strip().lower()

            if i != '' and not self.is_number(i) and i not in self.stop_words:
                n = n + 1
                if i in cleaned_dict:
                    cleaned_dict[i] = cleaned_dict[i] + 1
                else:
                    cleaned_dict[i] = 1
        return n, cleaned_dict

    def construct_dt_matrix(self, files, topK=200):
        #jieba.analyse.set_stop_words(self.stop_words_path)
        jieba.analyse.set_idf_path(self.idf_path)
        files = listdir(self.doc_dir_path)
        M = len(files)
        N = 1
        terms = {}
        dt = []
        for i in files:
            root = ET.parse(self.doc_dir_path + i).getroot()
            title = root.find('title').text
            body = root.find('body').text
            docid = int(root.find('id').text)
            tags = jieba.analyse.extract_tags(title + '。' + body, topK=topK, withWeight=True)
            # tags = jieba.analyse.extract_tags(title, topK=topK, withWeight=True)
            cleaned_dict = {}
            for word, tfidf in tags:
                word = word.strip().lower()
                if word == '' or self.is_number(word):
                    continue
                cleaned_dict[word] = tfidf
                if word not in terms:
                    terms[word] = N
                    N += 1
            dt.append([docid, cleaned_dict])
        dt_matrix = [[0 for i in range(N)] for j in range(M)]
        i = 0
        for docid, t_tfidf in dt:
            dt_matrix[i][0] = docid
            for term, tfidf in t_tfidf.items():
                dt_matrix[i][terms[term]] = tfidf
            i += 1

        dt_matrix = pd.DataFrame(dt_matrix)
        dt_matrix.index = dt_matrix[0]
        print('dt_matrix shape:(%d %d)' % (dt_matrix.shape))
        return dt_matrix

    def construct_k_nearest_matrix(self, dt_matrix, k):
        tmp = np.array(1 - pairwise_distances(dt_matrix[dt_matrix.columns[1:]], metric="cosine"))
        similarity_matrix = pd.DataFrame(tmp, index=dt_matrix.index.tolist(), columns=dt_matrix.index.tolist())
        for i in similarity_matrix.index:
            tmp = [int(i), []]
            j = 0
            while j < k:
                max_col = similarity_matrix.loc[i].idxmax(axis=1)
                similarity_matrix.loc[i][max_col] = -1
                if max_col != i:
                    tmp[1].append(int(max_col))  # max column name
                    j += 1
            self.k_nearest.append(tmp)

    # def construct_dt_matrix(self, files, topK=200):
    #     files = listdir(self.doc_dir_path)
    #     M = len(files)
    #     N = 1
    #     terms = {}
    #     dt = []
    #     dt_matrix =[]
    #     train_corpus= []
    #
    #     for i in files:
    #         root = ET.parse(self.doc_dir_path + i).getroot()
    #         title = root.find('title').text
    #         body = root.find('body').text
    #         docid = int(root.find('id').text)
    #         content = title + '。' + body
    #         seg_term = self.new_seg(content)
    #         train_corpus.append(seg_term)
    #         #ld,cleaned_dict = self.clean_list(seg_term)
    #
    #
    #     # vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    #     # transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    #     # tfidf = transformer.fit_transform(vectorizer.fit_transform(train_corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    #     # word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    #     # weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    #     # for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    #     #      print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
    #     #      for j in range(len(word)):
    #     #          if self.is_number(weight[i][j])  and float(weight[i][j]) > 0.0:
    #     #              tfidf
    #
    #
    #
    #
    #     #     cleaned_dict = {}
    #     #     for word, tfidf in seg_term:
    #     #         word = word.strip().lower()
    #     #         if word == '' or self.is_number(word):
    #     #             continue
    #     #         cleaned_dict[word] = tfidf
    #     #         if word not in terms:
    #     #             terms[word] = N
    #     #             N += 1
    #     #     dt.append([docid, cleaned_dict])
    #     # dt_matrix = [[0 for i in range(N)] for j in range(M)]
    #     # i = 0
    #     # for docid, t_tfidf in dt:
    #     #     dt_matrix[i][0] = docid
    #     #     for term, tfidf in t_tfidf.items():
    #     #         dt_matrix[i][terms[term]] = tfidf
    #     #     i += 1
    #     #
    #     # dt_matrix = pd.DataFrame(dt_matrix)
    #     # dt_matrix.index = dt_matrix[0]
    #     # print('dt_matrix shape:(%d %d)' % (dt_matrix.shape))
    #     return dt_matrix

    def construct_k_nearest_matrix(self, dt_matrix, k):
        tmp = np.array(1 - pairwise_distances(dt_matrix[dt_matrix.columns[1:]], metric="cosine"))
        similarity_matrix = pd.DataFrame(tmp, index=dt_matrix.index.tolist(), columns=dt_matrix.index.tolist())
        for i in similarity_matrix.index:
            tmp = [int(i), []]
            j = 0
            while j < k:
                max_col = similarity_matrix.loc[i].idxmax(axis=1)
                similarity_matrix.loc[i][max_col] = -1
                if max_col != i:
                    tmp[1].append(int(max_col))  # max column name
                    j += 1
            self.k_nearest.append(tmp)

    def gen_idf_file(self):
        files = listdir(self.doc_dir_path)
        n = float(len(files))
        idf = {}
        for i in files:
            root = ET.parse(self.doc_dir_path + i).getroot()
            title = root.find('title').text
            body = root.find('body').text
            seg_list = jieba.lcut(title + '。' + body, cut_all=False)
            seg_list = set(seg_list) - self.stop_words
            for word in seg_list:
                word = word.strip().lower()
                if word == '' or self.is_number(word):
                    continue
                if word not in idf:
                    idf[word] = 1
                else:
                    idf[word] = idf[word] + 1
        idf_file = open(self.idf_path, 'w', encoding='utf-8')
        for word, df in idf.items():
            idf_file.write('%s %.9f\n' % (word, math.log(n / df)))
        idf_file.close()

    def find_k_nearest(self, k, topK):
        self.gen_idf_file()
        files = listdir(self.doc_dir_path)
        dt_matrix = self.construct_dt_matrix(files, topK)
        self.construct_k_nearest_matrix(dt_matrix, k)
        self.write_k_nearest_matrix_to_db()

#TODO： 修改为pyhanlp，将jiba彻底替换掉
if __name__=='__main__':
    print('-----start time: %s-----' % (datetime.today()))
    filename = os.path.join(os.path.dirname(__file__), 'config.ini')
    rm = LearnRanking(filename, 'utf-8')
    rm.find_k_nearest(5, 25)
    print('-----finish time: %s-----' % (datetime.today()))
