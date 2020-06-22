#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:load_data.py
# @Author: Michael.liu
# @Date:2020/6/19 20:17
# @Desc: this code is ....

class LoadData(object):

    def __init__(self,filePath):
        self.datafile = filePath
        self.dataset = self.load_data()

    def load_data(self):
        dataset = []
        iLineNum =0
        with open(self.datafile,'r',encoding='utf8') as f:
            for line in f.readlines():
                iLineNum += 1
                line = line.strip().replace('±±±±',' ')
                dataset.append([word for word in line.split(' ') if 'nbsp' not in word and len(word) < 11])
        return dataset

if __name__ == "__main__":
    print("....start....")
    loadData = LoadData('../../../data/chapter7/xml_data_process.txt')
    dataset = loadData.load_data()

    for i in dataset:
        print(i)
    print(len(dataset))
   #print(dataset)