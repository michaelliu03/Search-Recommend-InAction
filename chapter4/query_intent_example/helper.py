#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:helper.py
# @Author: Michael.liu
# @Date:2020/5/14 15:19
# @Desc: this code is ....
import codecs
import numpy as np
from pyhanlp import *


def train_format(filein,fileout):
    fr = codecs.open(filein,'r',encoding='utf-8')
    fw = codecs.open(fileout, 'w',encoding='utf-8')
    # name,cat,cat_id,click
    for line in fr.readlines():
        line = line.strip().lower().replace("\n","")  # 简单标准化
        split_cell = line.split("\t")
        if len(split_cell) != 9 : continue
        else:
            name = split_cell[1]
            cat = split_cell[4]
            cat_id = split_cell[3]
            cat_id_i = int(cat_id)
            click  = split_cell[6]
            click_d = float(click)
            if name == "" and cat == "" and cat_id == "" and click == "":
                continue
            else:
                if cat_id_i == 3 and click_d > 0.1:
                    fw.write(seg_formate(name) + "\t" +  "__label__" + cat)
                    fw.write("\n")
            #print(name +"\t" + cat + "\t"+ cat_id + "\t" + click)
        #print(line)
    fr.close()
    fw.close()

def seg_formate(content):
    HanLP = SafeJClass('com.hankcs.hanlp.HanLP')
    ret = []
    seg_list = HanLP.segment(content)
    for term in seg_list:
        word = str(term.word).strip().lower()
        ret.append(word)
    seg_content = ' '.join(ret)
    return seg_content


def train_test_split_cn(filein,split_rate):
    s = []
    f = open(r"../../data/chapter4/intent/query_intent_train_used.csv","w",encoding='utf-8')
    g = open( r"../../data/chapter4/intent/query_intent_test_used.csv","w",encoding='utf-8')
    file = codecs.open(filein,'r', "utf-8")
    for line in file:
        s.append(line)
    np.random.shuffle(s)

    j = 0
    for i in s:
        j += 1
        if (j <= len(s) * split_rate):
            f.write(i)
        else:
            g.write(i)
    f.close()
    g.close()



if __name__ == "__main__":
   print("....start....")
   file_in = "../../data/chapter4/intent/query_intent_train.txt"
   file_out = "../../data/chapter4/intent/query_intent_train_out.csv"
   #train_format(file_in,file_out)
   train_test_split_cn(file_out,0.8)

   print(".....finished.....")