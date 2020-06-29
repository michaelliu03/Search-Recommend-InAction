#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @project : books_method
# @File : data_process.py
# @Time    : 2020/6/24 11:50
# @Author  : Zhaohy


choose_feature = '''


relevent

boolean model:body 2
boolean model:anchor 2
boolean model:title 2
boolean model:url 2
boolean model:whole document 2
as bin feature


covered query term number:body 10
covered query term number:anchor 7
covered query term number:title 10
covered query term number:url 9
covered query term number:whole document 10

as cat feature


sum of term frequency:body 479
sum of term frequency:anchor 32
sum of term frequency:title 65
sum of term frequency:url 15
sum of term frequency:whole document 489
min of term frequency:body 174
min of term frequency:anchor 17
min of term frequency:title 19
min of term frequency:url 5
min of term frequency:whole document 176
max of term frequency:body 366
max of term frequency:anchor 26
max of term frequency:title 55
max of term frequency:url 11
max of term frequency:whole document 388
mean of term frequency:body 1268
mean of term frequency:anchor 80
mean of term frequency:title 131
mean of term frequency:url 44
mean of term frequency:whole document 1283
variance of term frequency:body 7010
variance of term frequency:anchor 144
variance of term frequency:title 237
variance of term frequency:url 68
variance of term frequency:whole document 7348
PageRank 41707
SiteRank 43697
as normal continue values


'''




import  pandas as pd

import numpy as np
import matplotlib.pyplot as plt

def load_head(path):
    '''
    :param path: path of head explaination
    :return:  list  of head
    '''

    head = []
    count = 0
    with open(path,'r',encoding='utf8') as f:
        for line in f :

            if '\t' not in line: continue
            line = line.strip().split('\t')
            count += 1

            if len(line) != 2 : continue
            head.append(line[1])

    return head


def mico_data_process(path,head,key):

    df = pd.read_csv(path,header=0,sep=',')

    index_dict ={'body':1,'anchor':2,'title':3,'url':4,'whole document':5}
    count = 1

    feature_dict = {}
    select_feature = []
    for h in head :

        if h in choose_feature :
            cur = ''
            if ':' in h :
                t = h.split(':')
                cur = 'query_' + ''.join([i[0] for i in t[0].split()]) +'_' + str(index_dict[t[1]])

                if 'boolean' in h :
                    cur = cur + '_bin'
                if 'covered' in h :
                    cur = cur + '_cat'
            elif h == 'relevent':
                cur = 'target'
            else:
                cur = 'query_other_' + str(count)

            feature_dict[h] = cur
            select_feature.append(h)

    data = df[select_feature]
    data.rename(columns=feature_dict,inplace=True)


    #print(data)
    id_list = []



    for i in range(0 , len(data['target'])   ) :
        id_list.append(i)

    #print(len(data['target']) , len(id_list))

    data.insert(0,'id',id_list)

    if key !='train':
        data = data.drop(columns=['target'])
        print('drop target')

    #print(data)
    data.to_csv(path_or_buf='./data/fm_'+key+'.csv',sep='\t',index=False)



def main():

    head_path  = './data/feature_head.txt'

    head = load_head(head_path)
    print(len(head))

    train_path ='./data/format_train.txt'
    test_path = './data/format_test.txt'
    vali_path = './data/format_vali.txt'

    mico_data_process(train_path, head, 'train')
    mico_data_process(test_path,head,'test')
    mico_data_process(vali_path, head, 'vali')


