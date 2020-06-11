#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:kmeans_example.py
# @Author: Michael.liu
# @Date:2020/6/4 11:45
# @Desc: this code is ....

import codecs
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import  random
from matplotlib.font_manager import FontProperties

corpus_path = u"../../data/chapter4/cluster/xml_data_process.txt"

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
    news_data = pd.read_csv(corpus_path,sep='±±±±',encoding='utf-8')
    #print(rd.head(5))
    #print(type(rd))
    news_title = news_data['title'].tolist()
    news_content = news_data['content'].tolist()
    return news_title,news_content,news_data

def k_means(feature_matrix, num_clusters=10):
    km = KMeans(n_clusters=num_clusters,
                max_iter=10000)
    km.fit(feature_matrix)
    clusters = km.labels_
    return km, clusters

def get_cluster_data(clustering_obj, news_data,
                     feature_names, num_clusters,
                     topn_features=10):
    cluster_details = {}
    # 获取cluster的center
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
    # 获取每个cluster的关键特征
    # 获取每个cluster的书
    for cluster_num in range(num_clusters):
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        key_features = [feature_names[index] for index  in ordered_centroids[cluster_num, :topn_features]]
        cluster_details[cluster_num]['key_features'] = key_features

        news = news_data[news_data['Cluster'] == cluster_num]['title'].values.tolist()
        cluster_details[cluster_num]['content'] = news

    return cluster_details



def process():
    title,content,news_data = load_data()
    #Todo 去掉一些停用词
    filter_content = content
    vectorizer, feature_matrix = build_feature_matrix(filter_content,
                                                      feature_type='tfidf',
                                                      min_df=0.2, max_df=0.90,
                                                      ngram_range=(1, 2))
    #print(feature_matrix.shape)
    # 获取特征名字
    feature_names = vectorizer.get_feature_names()
    #print(feature_names)
    # 打印某些特征
    print(feature_names[:10])
    num_clusters = 10
    km_obj, clusters = k_means(feature_matrix=feature_matrix,
                               num_clusters=num_clusters)

    news_data['Cluster'] = clusters
    c = Counter(clusters)
    print(c.items())

    # 取 cluster 数据
    cluster_data = get_cluster_data(clustering_obj=km_obj,
                                    news_data=news_data,
                                    feature_names=feature_names,
                                    num_clusters=num_clusters,
                                    topn_features=5)




if __name__ == '__main__':
    print("start>>>>>>")
    process()
    print(">>>>>>>>end")




