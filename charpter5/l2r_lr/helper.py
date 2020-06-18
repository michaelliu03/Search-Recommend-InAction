#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:helper.py
# @Author: Michael.liu
# @Date:2020/6/17 14:51
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
            line = line.strip().split('\t')
            if len(line) != 2 : continue
            head.append(line[1])

    return head

def data_process(root,path):
    '''

    :param path: path to load mico data
    :return: pandas dataframe data for model
    '''

    head = ['index','qid']
    for i in range(1,137):
        head.append('f'+str(i))

    o = open(root + 'format_' + path,'w',encoding='utf8')
    o.write(','.join(head) + '\n')
    count = 0
    with open(root + path,'r',encoding='utf8') as f:
        for line in f :
            line = line.strip().split()
            for i in range(1,len(line)):
                line[i] = line[i].split(':')[1]
                line[i] = str(float(line[i]))
            o.write(','.join(line) + '\n')

# 数据预处理
def pre_process_data():
    root = './'
    train_path = 'train.txt'
    data_process(root,train_path)

    test_path = 'test.txt'
    data_process(root,test_path)

    vali_path = 'vali.txt'
    data_process(root,vali_path)