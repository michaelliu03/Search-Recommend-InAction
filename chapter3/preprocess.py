#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:preprocess.py
# @Author: Michael.liu
# @Date:2019/2/12
# @Desc: 这个类主要处理采集来的新闻语料，新闻语料是在第2章的讲解过程中采集到的
import os
from os import listdir
import logging
from pyhanlp import *
import codecs
import re
import pandas as pd
import numpy as np


import xml.etree.ElementTree as ET
logger = logging.getLogger(__name__)

#定义一个root，需要根据root拼整个文件件的完整路径
dir_root =  os.path.dirname("D:\\coding\\self-project\\Search-Recommend-InAction\\Search-Recommend-InAction\\data\\charpter2\\news\\")

def new_seg(content):
    ret = []
    seg_list = HanLP.segment(content)
    for term in seg_list:
        word = str(term.word)
        word_nature = str(term.nature)
        #print(word_nature)
        if word_nature == 'nr' or word_nature == 'ns' or word_nature == 'nt':
            word_new = term.word
            word_nature = str(term.nature)
            term_new = word_new + '/' + word_nature
            term_new = term_new + " "
            ret.append(term_new)
        else:
            word_new =  term.word
            term_new = word_new + '/' + 'o'
            term_new = term_new + " "
            ret.append(term_new)
    return ret


def del_corpus(filepath):
    logger.info("start del_corpus >>>>>>>")
    output_data = codecs.open('train1.txt','w','utf-8')
    files = listdir(filepath)
    for i in files:
        file_path = dir_root + "\\"+i
        root = ET.parse(file_path).getroot()
        title = root.find('title').text.replace('\t','').replace('\r','').replace('\n','')
        body = root.find('body').text.replace('\t','').replace('\r','').replace('\n','')
        #docid = int(root.find('id').text)
        #date_time = root.find('datetime').text
        content = "".join(title + '。' + body)
        seg_py_list = new_seg(content)
        for i in seg_py_list:
            output_data.write(i)
        output_data.write('\n')
    output_data.close()

def wordtag():
    input_data = codecs.open('train1.txt', 'r', 'utf-8')
    output_data = codecs.open('wordtag.txt', 'w', 'utf-8')
    row =0
    for line in input_data.readlines():
        # line=re.split('[，。；！：？、‘’“”]/[o]'.decode('utf-8'),line.strip())
        line = line.strip().split()

        if len(line) == 0 :
            continue
        for word in line:
            word = word.split('/')
            try:
                if word[1] != 'o':
                    if len(word[0]) == 1:
                        output_data.write(word[0] + "/B_" + word[1] + " ")
                    elif len(word[0]) == 2:
                        output_data.write(word[0][0] + "/B_" + word[1] + " ")
                        output_data.write(word[0][1] + "/E_" + word[1] + " ")
                    else:
                        try:
                            output_data.write(word[0][0] + "/B_" + word[1] + " ")
                            for j in word[0][1:len(word[0]) - 1]:
                                output_data.write(j + "/M_" + word[1] + " ")
                            output_data.write(word[0][-1] + "/E_" + word[1] + " ")
                        except:   #这是一个坑，没有找具体的原因
                            continue
                else:
                    for j in word[0]:
                        output_data.write(j + "/o" + " ")
            except:   #这是一个坑，没有找具体的原因
                continue
        output_data.write('\n')

    input_data.close()
    output_data.close()
    logger.info("this is finished!")

if __name__ == "__main__":
    print("......begin......")
    #filepath = os.path.dirname("D:\\coding\\self-project\\Search-Recommend-InAction\\Search-Recommend-InAction\\data\\charpter2\\news\\")
    #del_corpus(filepath)
    wordtag()

