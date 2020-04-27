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
import collections
import pandas as pd
import numpy as np
import json

import xml.etree.ElementTree as ET
logger = logging.getLogger(__name__)

#定义一个root，需要根据root拼整个文件件的完整路径
dir_root =  os.path.dirname("D:\\coding\\self-project\\Search-Recommend-InAction\\Search-Recommend-InAction\\data\\charpter2\\news-2020-04-26-part2-2020-04-26-part1-2020-04-26-2020-04-22-part2-2020-04-22-part1-2020-04-20\\")

def check_contain_chinese(check_str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False



def new_seg(content):
    ret = []
    seg_list = HanLP.segment(content)
    for term in seg_list:
        word = str(term.word)
        word_nature = str(term.nature)

        # 去所有的非中文字符
        if check_contain_chinese(word) is not True:
            continue

        if word_nature == 'nr' or word_nature == 'ns' or word_nature == 'nt':
            word_new = term.word
            word_nature = str(term.nature)
            term_new = word_new + '/' + word_nature
            term_new = term_new + " "
            ret.append(term_new)
        else:
            if word.strip() == ' ':
                continue
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

datas = list()
labels = list()
linedata = list()
linelable = list()

tag2id = {'' :0,
        'B_ns' :1,
        'B_nr' :2,
        'B_nt' :3,
        'M_nt' :4,
        'M_nr' :5,
        'M_ns' :6,
        'E_nt' :7,
        'E_nr' :8,
        'E_ns' :9,
         'o': 0}

id2tag = {  0:'' ,
            1:'B_ns' ,
            2:'B_nr' ,
            3:'B_nt' ,
            4:'M_nt' ,
            5:'M_nr' ,
            6:'M_ns' ,
            7:'E_nt' ,
            8:'E_nr' ,
            9:'E_ns' ,
            10: 'o'}

def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

#输出词表
def output_vocabulary(abc,filename):
    with open(filename,'w') as outfile:
        json.dump(abc,outfile,indent=4)

max_len =  500

def format_corpus(filepath):
    input_data = codecs.open(filepath, 'r', 'utf-8')
    for line in input_data.readlines():
        line = re.split('[，。；！：？、‘’“”]/[o]', line.strip())
        for sen in line:
            sen = sen.strip().split()
            if len(sen) == 0:
                continue
            linedata = []
            linelabel = []
            num_not_o = 0
            for word in sen:
                word = word.split('/')
                # 这里有一个坑
                try:
                    linedata.append(word[0])
                    linelabel.append(tag2id[word[1]])
                except:
                    continue

                if word[1] != 'o':
                    num_not_o += 1
            if num_not_o != 0:
                datas.append(linedata)
                labels.append(linelabel)

    input_data.close()
    print(len(datas))
    print(len(labels))


    all_words = flatten(datas)
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()
    set_words = sr_allwords.index
    set_ids = range(1, len(set_words) + 1)
    word2id = pd.Series(set_ids, index=set_words)
    word2id.to_json('vocabulary_word2id.json')

    id2word = pd.Series(set_words, index=set_ids)
    word2id["unknown"] = len(word2id) + 1

    def X_padding(words):
       ids = list(word2id[words])
       if len(ids) >= max_len:
           return ids[:max_len]
       ids.extend([0] * (max_len - len(ids)))

    def y_padding(ids):
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids)))
        return ids

    df_data = pd.DataFrame({'words':datas,'tags':labels},index=range(len(datas)))
    df_data['x'] = df_data['words'].apply(X_padding)
    df_data['y'] = df_data['tags'].apply(y_padding)
    x = np.asarray(list(df_data['x'].values))
    y = np.asarray(list(df_data['y'].values))

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=43)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=43)

    print('Finished creating the data generator.')

    import pickle
    import os
    with open('../dataMSRA.pkl', 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)
        pickle.dump(x_valid, outp)
        pickle.dump(y_valid, outp)
    print('** Finished saving the data.')

if __name__ == "__main__":
    print("......begin......")
    filepath = os.path.dirname("D:\\coding\\self-project\\Search-Recommend-InAction\\Search-Recommend-InAction\\data\\charpter2\\news-2020-04-26-part2-2020-04-26-part1-2020-04-26-2020-04-22-part2-2020-04-22-part1-2020-04-20\\")
    del_corpus(filepath)
    #wordtag()
    #meragefilepath = "D:\\coding\\self-project\\Search-Recommend-InAction\\Search-Recommend-InAction\\data\\charpter2\\format_train\\wordtag.txt"
    #format_corpus(meragefilepath)
    print("finished!!")
