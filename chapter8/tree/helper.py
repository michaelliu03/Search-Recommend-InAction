#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:helper.py
# @Author: Michael.liu
# @Date:2020/6/30 13:19
# @Desc: this code is ....



def load_head(path):
    '''
    :param path: path of head explaination
    :return:  list  of head
    '''

    head = []
    with open(path,'r',encoding='utf8') as f:
        for line in f :
            if '\t' not in line: continue
            line = line.replace(':','=')
            line = line.strip().split('\t')
            if len(line) != 2 : continue
            head.append(line[1])

    return head