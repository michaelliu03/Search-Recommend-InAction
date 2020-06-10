#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:utils.py
# @Author: Michael.liu
# @Date:2020/6/3 17:49
# @Desc: this code is ....
import pandas as pd

# tf_preprocess
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import pickle as pkl
import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix

def clean_df(df,training=True):
    df = df.drop(
        ['site_id', 'app_id','device_id','device_ip','site_domain','site_category','app_domain', 'app_category'],
        axis=1, inplace=True)

    if training:
        df = df.drop(['id'],axis=1)
    return df

def load_df(filename, training=True,**csv_options):
    df = pd.read_csv(filename,header=0,**csv_options)

    #df = clean_df(df,training=training)
    df = select_df(df,training=training)
    return df

def select_df(df,training=True):
    features = pd.read_csv('feature.csv')
    x_columns = features.head(30)['feature'].tolist()
    df = df[x_columns]
    if training:
        df = df.drop(['id'], axis=1)
    return df