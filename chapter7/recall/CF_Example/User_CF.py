#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:User_CF.py
# @Author: Michael.liu
# @Date:2020/6/19 13:37
# @Desc: this code is ....
import argparse
import time
from math import  sqrt,log
from operator import  add

from pyspark import SparkContext

def read_data(file_path,sparkContext):
    '''

    :param file_path:
    :param sparkContext:
    :return: RDD(userID,goodsID)
    '''
    data_rdd = sparkContext.textFile(file_path,use_unicode=False)\
        .map(lambda line:line.strip())\
        .map(lambda line:line.split(","))\
        .map(lambda line:(int(line[0]),int(line[1])))

    (train_rdd,test_rdd) = data_rdd.randomSplit(weights=[0.75,0.25],seed=0)

    print("read data finished !")
    return  train_rdd,test_rdd

def calc_user_sim(train_rdd):
    #Get Item-User inverse tabel
    print("building item-user inverse table")


    good2users = train_rdd \
        .map(lambda (user, goods): (goods, user)) \
        .groupByKey(numPartitions=40) \
        .map(lambda (goods, user_list): (goods, [u for u in user_list]))

    # count popularity
    goods_popular = good2users\
        .map(lambda (goods,user_list):(goods,len(user_list)))

    all_goods_count = good2users.count()

    user_co_rated_matrix = good2users\
        .map(lambda (goods,user_list):get_user_sim_matrix(goods,user_list))\
        .flatMap(lambda uv_list:uv_list)\
        .map(lambda (u,v):((u,v),1))\
        .reduceByKey(add,numPartitions=40)
    print('build goods-users inverse table succ')

    view_num_map = train_rdd\
        .map(lambda (user,goods):(user,1))\
        .reduceByKey(add,numPartitions=40)\
        .collectAsMap()

    user_sim_matrix = user_co_rated_matrix\
        .map(lambda ((u,v),count):((u,v).count
          /sqrt(view_num_map[u]*view_num_map[v])))
    print("calculate user user_sim_matrix ")
    return user_sim_matrix,goods_popular,all_goods_count


def get_user_sim_matrix(goods,user_list):
    uv_list = []
    user_list.sort()
    for u  in user_list:
        for v in user_list:
            if u==v:
                continue
            uv_list.append((u,v))
    return uv_list





if __name__ == '__main__':
    print("....begin....")
    start_time = time.time()
    parser = argparse.ArgumentParser(description='UserCF Spark',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_sim_goods')
    parser.add_argument('--n_rec_goods')
    parser.add_argument('--input',default=None,help='Input Data')
    parser.add_argument('--master',default="local[20]",help="Spark Master")

    parser.set_defaults(verbose =False)

    args = parser.parse_args()
    sc = SparkContext(args.master,'UserCF Spark Version')

    train_set,test_set = read_data(file_path=args.input,sparkContext=sc)
    user_similarity_matrix, goods_popular_count, goods_total_count = calc_user_sim(train_rdd=train_set)
