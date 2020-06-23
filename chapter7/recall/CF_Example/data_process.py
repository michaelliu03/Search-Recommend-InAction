#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:data_process.py
# @Author: Michael.liu
# @Date:2020/6/19 20:06
# @Desc: this code is ....
import pandas as pd
import numpy as np
import argparse

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


if __name__ == "__main__":
    print("....begin....")
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", help="file path of model")
    args = parser.parse_args()
    data_preprocess(args.filepath)


    print("....end....")
