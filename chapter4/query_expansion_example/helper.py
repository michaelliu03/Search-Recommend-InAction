#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:helper.py
# @Author: Michael.liu
# @Date:2020/5/12 14:58
# @Desc: this code is ....
import codecs
import os
from pyhanlp import *

def load_d_cut(inputfile,outfilepath):
    fr = codecs.open(inputfile,'r',encoding='utf-8')
    for line in fr.readline():
        print(line)



    fr.close()

load_d_cut("","./outfile.csv")