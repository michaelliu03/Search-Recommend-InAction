#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:query_dictionary.py
# @Author: Michael.liu
# @Date:2020/5/9 16:12
# @Desc: this code is ....



import codecs
import jpype
from  jpype import *
import os
from  pypinyin import *








def build_model():
    word_dict = {}
    count = 0
    for line in open('./SogouLabDic.dic.utf8','r',encoding='utf-8'):
        count += 1
        print(count)
        line = line.strip().split('\t')
        word = line[0]
        word_count = line[1]
        word_pinyin = ','.join(lazy_pinyin(word))
        if word_pinyin not in word_dict:
            word_dict[word_pinyin] = word + '_' + word_count
        else:
            word_dict[word_pinyin] += ';' + word + '_' + word_count

    data = {}
    for pinyin, words in word_dict.items():
        tmp = {}
        for word in words.split(';'):
            word_word = word.split('_')[0]
            word_count = int(word.split('_')[1])
            tmp[word_word] = word_count
        data[pinyin] = tmp


    f = open('query_correct.model', 'w',encoding='utf-8')
    f.write(str(data))
    f.close()



if __name__ =="__main__":
    print("......start......")
    build_model()
    print("......finished......")

