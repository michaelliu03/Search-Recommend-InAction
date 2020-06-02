#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:data_util.py
# @Author: Michael.liu
# @Date:2020/6/2 13:32
# @Desc: this code is ....
import os
import time
import numpy as np
import json

QUERY_IDS = 'query_ids'
FEATURES = 'features'
LABEL_LIST = 'label_list'
DATA_PATH = "data/Fold1/"
#print(DATA_PATH)

def data_convert(type):
    data_path =DATA_PATH+'\\'+type+'.txt' #os.path.join('',)
    print(data_path)
    label_list = list()
    features = list()
    current_row = 0
    with open(data_path,'r') as f:
        for line in f:
            #print(line)
            current_row +=1
            q2 = line.split(" ")
            label_list.append(q2[0])
            del q2[0]
            d = ':'.join(map(str,q2))
            e =  d.split(":")
            features.append(e[1::2])
            if  current_row % 50000 ==0:
                print('row %d - %f seconds' % (current_row,time.time() - start_time))
    print('Done loading data - %f seconds' % (time.time() - start_time))
    label_list = np.asarray(label_list,dtype=int)

    features = np.asarray(features, dtype=float)
    query_ids = np.asarray(features[:, 0], dtype=int)
    features = features[:, 1:]
    np_file_directory = 'data/np_' + type + '_files'
    np.save(os.path.join(np_file_directory, LABEL_LIST), label_list)
    with open(os.path.join(np_file_directory, LABEL_LIST), "w") as dump_f_1:
        json.dump(label_list.tolist(),dump_f_1)
    np.save(os.path.join(np_file_directory, FEATURES), features)
    with open(os.path.join(np_file_directory, FEATURES),"w") as dump_f_2:
        json.dump(features.tolist(),dump_f_2)

    np.save(os.path.join(np_file_directory, QUERY_IDS), query_ids)
    with open(os.path.join(np_file_directory, QUERY_IDS), "w") as dump_f:
         json.dump(query_ids.tolist(),dump_f)






if __name__ == '__main__':
    print("...start...")
    start_time = time.time()
    data_convert('train')
    print("...end...")