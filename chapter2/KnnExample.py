#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:Segment.py
# @Author: Michael.liu
# @Date:2019/2/12
# @Desc: NLP Segmentation ToolKit - Hanlp Python Version

import math

#   邻近算法，或者说K最近邻(kNN，k-NearestNeighbor)分类算法是数据挖掘分类技术中最简单的方法之一。
#   所谓K最近邻，就是k个最近的邻居的意思，说的是每个样本都可以用它最接近的k个邻居来代表。
#   Cover和Hart在1968年提出了最初的邻近算法。
#   KNN是一种分类(classification)算法，它输入基于实例的学习（instance-based learning），属于懒惰学习（lazy learning）
#   即KNN没有显式的学习过程，也就是说没有训练阶段，
#   数据集事先已有了分类和特征值，待收到新样本后直接进行处理。
#   与急切学习（eager learning）相对应。
#　 KNN是通过测量不同特征值之间的距离进行分类。

def dis(a,b):
    '''

    :param a:
    :param b:
    :return:
    '''
    if len(a) != len(b):
        return -1
    tem = 0
    for i in range(len(a)):
        tem = tem + math.pow((a[i]-b[i]),2)
    return math.sqrt(tem)

def KNN(train_list, test_list, K):
    train_res = {}
    point_index = 0
    for point in train_list:
        neighbor_dic = {}
        # 计算每一个测试集合中的点与当前点的距离
        ner_index = 0
        for neighbor in test_list:
            distance = dis(point,neighbor)
            neighbor_dic[ner_index] = distance
            ner_index +=1
        tem =0
        votes ={}
        #按照距离由小到大至K投票，将票数最高的统计为当前类别
        for k,v in sorted(neighbor_dic.items(),key=lambda k:k[1],reverse = False):
            tem +=1
            if tem > K : break
            class_type = cluster_type(test_list[k])
            if class_type in votes:
                votes[class_type] +=1
            else:
                votes[class_type]  =1
        train_res[point_index] = sorted(votes.items(),key =lambda k:k[1],reverse=True)[0][0]
        point_index +=1
    return train_res


#
def cluster_type(arr):
    if arr == [3,3] or arr == [4,4] : return 0
    else : return 1


def main():
    train_list = [[3, 4], [12, 5], [4, 7], [11, 7], [20, 11], [15, 15]]
    test_list = [[3, 3], [10, 10], [4, 4], [11, 11], [14, 14]]

    K = 3
    re = KNN(train_list, test_list, K)
    for i in re:
        print(train_list[i], 'belong to cluster ', re[i])

if __name__ =='__main__':
    main()