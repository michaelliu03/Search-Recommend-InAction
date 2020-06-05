#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:kmeans_example.py
# @Author: Michael.liu
# @Date:2020/6/4 11:45
# @Desc: this code is ....


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.cluster import k_means_
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import  random

corpus_path = u""

def build_feature_matrix(document,feature_type='frequency',ngram_range=(1,1),min_df=0.0,max_df=1.0):
    feature_type = feature_type.lower().strip()
    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True,max_df=max_df,ngram_range=ngram_range)
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary= False,min_df = min_df,max_df=max_df,ngram_range=ngram_range)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise Exception("Wrong feature type entered.Possible values:'binary','frequency','tfidf'")
    feature_matrix = vectorizer.fit_transform(document).astype(float)
    print(feature_matrix)
    return  vectorizer,feature_matrix

def load_data():
    rd = pd.read_csv(corpus_path)



def process():
    news_data = load_data()






if __name__ == '__main__':
    print("start>>>>>>")
    process()
    print(">>>>>>>>end")




