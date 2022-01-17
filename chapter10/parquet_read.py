#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 
# @desc :
import pandas as pd
from pandas import  read_parquet

def verify_product(path):
    data = read_parquet(path)
    print(data.count())
    data.head()

def verify_user(path):
    data = read_parquet(path)
    print(data.count())



if __name__ == '__main__':
    path1 = "../chapter10/myCollaborativeFilter/data/product"
    verify_product(path1)