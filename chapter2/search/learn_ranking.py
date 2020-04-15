#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:learn_ranking.py
# @Author: Michael.liu
# @Date:2019/2/12
# @Desc:

from os import listdir
import xml.etree.ElementTree as ET
from pyhanlp import *
import sqlite3
import configparser
from datetime import *
import math
import pandas as pd
import numpy as np

from sklearn.metrics import pairwise_distances

class LearnRanking:
    stop_words = set()
    k_nearest = []

    config_path =''
    config_encoding = ''

    doc_dir_path = ''
    doc_encoding = ''
    stop_words_path = ''
    stop_words_encoding= ''
    idf_path= ''
    db_path = ''

    def __init__(self,config_path,config_encoding):
        print()


if __name__=='__main__':
    print("this is learn_ranking")
