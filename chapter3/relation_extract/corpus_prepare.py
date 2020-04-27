#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:corpus_prepare.py
# @Author: Michael.liu
# @Date:2020/4/23 13:48
# @Desc: this code is 准备抽取关系的语料

from os import listdir
import logging
import codecs
import re
import collections
import pandas as pd
import numpy as np
import json
from pyhanlp import *
import xml.etree.ElementTree as ET
from .utils import *


dir_root = "D:\\coding\\self-project\\Search-Recommend-InAction\\Search-Recommend-InAction\\data\\charpter2\\news-2020-04-26-part2-2020-04-26-part1-2020-04-26-2020-04-20\\"
stop_words = set()

def read_stopwords(file_path):
    f = open(file_path, encoding='utf-8')
    words = f.read()
    stop_words = set(words.split('\n'))
    return stop_words

def opt_corpus(in_file_path,out_file_path):
    output_data = codecs.open(out_file_path, 'w', 'utf-8')
    files = listdir(in_file_path)
    print(files)
    for i in files:
        file_path = dir_root + "\\" + i
        root = ET.parse(file_path).getroot()
        title = root.find('title').text.replace('\t', '').replace('\r', '').replace('\n', '')
        body = root.find('body').text.replace('\t', '').replace('\r', '').replace('\n', '')
        content = title + "。" + body
        #print(title)
        output_data.write(content)
        output_data.write("\n")
    output_data.close()

def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

corpus_dic ={}
corpus_sentences ={}
corpus_sentences_main ={}


def extract_main(filename):
    input_data = codecs.open('./sentence_corpus.txt', 'r', 'utf-8')
    out_data = codecs.open('./sentence_corpus_parsing.txt', 'w', 'utf-8')
    n = 0
    ret = []
    for line in input_data:
        n += 1
        corpus_id = line.split("\t")[0]
        line_id = line.split("\t")[1]
        content = line.split("\t")[2]
        ret.append(pasing_ch(content, corpus_id, line_id))
    input_data.close()
    for item in ret:
        out_data.write(item)
        out_data.write("\n")
    print("finished!")
    out_data.close()

def pasing_ch(content,corpusid,numid):#传入文本的ID，行数，及文本
    analysis_obj=""
    sentences_parsing = HanLP.parseDependency(content)
    for word in sentences_parsing.iterator():  # 通过dir()可以查看sentence的方法
        if word.DEPREL == "主谓关系" or word.DEPREL == "动宾关系":
           #print("%s --(%s)--> %s" % (word.LEMMA, word.DEPREL, word.HEAD.LEMMA))
            analysis_obj = corpusid + "\t" + numid + "\t" + word.LEMMA + "↑"+ word.HEAD.LEMMA + "↑" + word.DEPREL
    return analysis_obj

if __name__ =="__main__":
    print("begin......")
    # step 1
    #filepath = os.path.dirname("D:\\coding\\self-project\\Search-Recommend-InAction\\Search-Recommend-InAction\\data\\charpter2\\news-2020-04-26-part2-2020-04-26-part1-2020-04-26-2020-04-20\\")
    #savefilename= 'news_corpus.txt'
    #opt_corpus(filepath)
    # step 2
    read_stopwords('./stop_words.txt')
    extract_main('./news_corpus.txt')
    print("finished!")

