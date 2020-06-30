#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:data_process.py
# @Author: Michael.liu
# @Date:2020/6/19 20:06
# @Desc: this code is ....
import pandas as pd
import numpy as np
import argparse
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import pyspark.sql.functions as F
from pyspark.sql.types import *

def data_preprocess(filepath):
   fw = open("./UserBehavior_pre.csv", "w", encoding='utf8')
   fw.write("userid" + "\t" + "goodsid" + "\t" + "goodsclassid" + "\t" + "bev" + "\t" + "time"+"\n")
   with open(filepath,'r',encoding='utf8') as f:
      lines = f.readlines()
      for line in lines:
          lineCell = line.strip().split(",")
          status = lineCell[3].lower()
          if status == "buy":
              status = "5"
          elif status == 'cart':
              status = "3"
          elif status == 'pv':
              status = "2"
          elif status == 'fav':
              status = "1"
          else:
              status = "0"
          fw.write(lineCell[0] + "\t"+ lineCell[1] + "\t" + lineCell[2] + "\t" + status + "\t" + lineCell[4] + "\n")

   fw.close()


def rate_score(str):
    score = 0.0
    if str == "buy":
        score = 5.0
    elif str == "cart":
        score = 3.0
    elif str == "pv":
        score = 2.0
    elif str == "fav":
        score = 1.0
    else:
        score = 0.0
    return score

def preprocess():
    spark = SparkSession.\
        builder.\
        appName("data_preprocess").\
        getOrCreate()
    ratings = spark.\
        read.\
        load("./UserBehavior.csv", format="csv", sep=",", inferSchema="true", header="false")
    # convert to a UDF Function by passing in the function and return type of function
    udfsomefunc = F.udf(rate_score, DoubleType())
    rating_new = ratings.withColumn("rating_new", udfsomefunc("rating"))
    rating_new.show(20)
    rating_new_2 = rating_new.select([c for c in rating_new.columns if c in ['_c0', '_c1', '_c2', '_c4', 'rating_new']])
    rating_new_2.\
        repartition(1).\
        write.csv("file:///data1/jupyter/liuyu5/c7/Userbehavior_Pre.csv",mode="overwrite")

if __name__ == "__main__":
    print("....begin....")
    preprocess()
    print("....end....")
