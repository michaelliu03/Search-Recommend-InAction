#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:User_CF.py
# @Author: Michael.liu
# @Date:2020/6/19 13:37
# @Desc: this code is ....
import argparse
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import pyspark.sql.types as T

def UserCF():
    print("...begin...")



if __name__ == '__main__':
    print("....begin....")
    spark = SparkSession \
        .builder \
        .appName("User_CF") \
        .getOrCreate()
    lines = spark.read.text("").rdd


