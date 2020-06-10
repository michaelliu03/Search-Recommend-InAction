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

def plot_clusters(num_clusters, feature_matrix,
                  cluster_data, news_data,
                  plot_size=(16, 8)):
    # generate random color for clusters
    def generate_random_color():
        color = '#%06x' % random.randint(0, 0xFFFFFF)
        return color

    # define markers for clusters
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
    # build cosine distance matrix
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    # dimensionality reduction using MDS
    mds = MDS(n_components=2, dissimilarity="precomputed",
              random_state=1)
    # get coordinates of clusters in new low-dimensional space
    plot_positions = mds.fit_transform(cosine_distance)
    x_pos, y_pos = plot_positions[:, 0], plot_positions[:, 1]
    # build cluster plotting data
    cluster_color_map = {}
    cluster_name_map = {}
    for cluster_num, cluster_details in cluster_data[:].items():
        # assign cluster features to unique label
        cluster_color_map[cluster_num] = generate_random_color()
        cluster_name_map[cluster_num] = ', '.join(cluster_details['key_features'][:5]).strip()
    # map each unique cluster label with its coordinates and books
    cluster_plot_frame = pd.DataFrame({'x': x_pos,
                                       'y': y_pos,
                                       'label': news_data['Cluster'].values.tolist(),
                                       'title': news_data['title'].values.tolist()
                                       })
    grouped_plot_frame = cluster_plot_frame.groupby('label')
    # set plot figure size and axes
    fig, ax = plt.subplots(figsize=plot_size)
    ax.margins(0.05)
    # plot each cluster using co-ordinates and book titles
    for cluster_num, cluster_frame in grouped_plot_frame:
        marker = markers[cluster_num] if cluster_num < len(markers) \
            else np.random.choice(markers, size=1)[0]
        ax.plot(cluster_frame['x'], cluster_frame['y'],
                marker=marker, linestyle='', ms=12,
                label=cluster_name_map[cluster_num],
                color=cluster_color_map[cluster_num], mec='none')
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off',
                       labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off',
                       labelleft='off')
    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), fancybox=True,
              shadow=True, ncol=5, numpoints=1, prop=fontP)
    # add labels as the film titles
    for index in range(len(cluster_plot_frame)):
        ax.text(cluster_plot_frame.ix[index]['x'],
                cluster_plot_frame.ix[index]['y'],
                cluster_plot_frame.ix[index]['title'], size=8)
        # show the plot
    plt.show()

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

    plot_clusters(num_clusters=num_clusters,
                  feature_matrix=feature_matrix,
                  cluster_data=cluster_data,
                  news_data=news_data,
                  plot_size=(16, 8))



if __name__ == '__main__':
    print("start>>>>>>")
    process()
    print(">>>>>>>>end")




