#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:helper.py
# @Author: Michael.liu
# @Date:2020/5/12 14:58
# @Desc: this code is ....
import codecs
import os
from pyhanlp import *
import xml.etree.ElementTree as ET

content_list = []

def read_file_list(inpufile,seg):
    dir_list = []
    file_list = []

    root = os.path.abspath(inpufile)
    print(root)
    dir_list = os.listdir(inpufile)
    for dir in dir_list:
        file_root = os.path.join(root + os.path.sep + dir)
        file_list = os.listdir(file_root)
        print(file_list)

        for i in file_list:
            filepath = os.path.join(file_root+os.path.sep+i)
            xml_root = ET.parse(filepath).getroot()
            title = xml_root.find('title').text
            body = xml_root.find('body').text
            content = title + seg + body #这里分割
            content_list.append(content)



def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
       return False

def new_seg(content,stop_words):
    HanLP = SafeJClass('com.hankcs.hanlp.HanLP')
    ret = []
    seg_list = HanLP.segment(content)
    for term in seg_list:
        word = str(term.word)
        word = str(term.word).strip().lower()
        if word == '\r' or word == '\r\n' or word =='\n' or len(word) <=1:
            continue
        if word != '' and word not in stop_words and  not is_number(word):
            ret.append(word)
    seg_content = ' '.join(ret)
    return seg_content

def load_d_cut(content_list,out_file):
    # 读停用词典
    fw = codecs.open(out_file,'w',encoding='utf-8')
    f = open('./stop_words.txt','r', encoding='utf-8')
    words = f.read()
    stop_words = set(words.split('\n'))
    for item in content_list:
        #print(item)
        seg_content = new_seg(item,stop_words)
        fw.write(str(seg_content))
        fw.write("\n")

    fw.close()


if __name__ == "__main__":
    print("......start.....")
    file_path = "../../data/chapter2/"
    read_file_list(file_path,'↑')
    load_d_cut(content_list,'./title_content_seg.txt')
    print("......finished!.......")
    #load_d_cut("","./outfile.csv")