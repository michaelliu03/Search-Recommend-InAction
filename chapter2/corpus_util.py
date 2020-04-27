#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:corpus_util.py
# @Author: Michael.liu
# @Date:2020/4/26 17:20
# @Desc:  处理爬虫下载的代码，用于文本分析

import os
from os import listdir
import xml.etree.ElementTree as ET
from pyhanlp import *
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import lda
import codecs



stop_words = set()
filter_term_nature = ('n', 'nr', 'ns', 'nt', 'nb', 'v',
                      'd','a','ad','nb','nba','nbc','nbp',
                      'nf','ng','nh','nhd','nhm','ni','nic',
                      'nis','nit','nl','nm','nmc','nn','nnd',
                      'nnt','nr1','nr2','nrf','')

def read_stop_words(filepath):
    f = open(filepath, 'r',encoding='utf-8')
    words = f.read()
    stop_words = set(words.split('\n'))
    #print(stop_words)




def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_all_files(dir):
    files_ = []
    list = os.listdir(dir)
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        if os.path.isdir(path):
            files_.extend(get_all_files(path))
        if os.path.isfile(path):
            files_.append(os.path.abspath(path))
    return files_

#需要同时过滤一下非法字符和词性
def segment_py(content):
    ret_words = []
    seg_list = HanLP.segment(content)

    for term in seg_list:
        term_word = str(term.word)
        term_nature = str(term.nature)
        if term_word == " ":
            continue
        else:

           if  is_number(term_word) ==True and term_word not in stop_words:
                continue
           if term_nature not in filter_term_nature:
                continue
        ret_words.append(term_word)

    return ret_words





def corpusHandler(dirname,filename):
    output_data = codecs.open(filename,'w',encoding='utf-8')
    files_ = get_all_files(dirname)
    #ret = []
    for file in files_:
        root = ET.parse(file).getroot()
        title = root.find('title').text
        body = root.find('body').text
        content = title + "。"+ body
        seg_list = segment_py(content)  # list
        new_line = ' '.join(seg_list)
        output_data.write(new_line)
        output_data.write("\n")

    output_data.close()




if __name__ =="__main__":
    print("......start......")
    # step 1: 读取停用词
    stop_words_filepath = u'./stop_words.utf8'
    read_stop_words(stop_words_filepath)
    # step 2: 读取一个文件夹下所有的文件
    corpus_file_path = u'./data/'
    new_file_out = u'./lda_prepare.csv'
    corpusHandler(corpus_file_path,new_file_out)




    print("......finished......")





