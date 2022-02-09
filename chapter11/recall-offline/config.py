#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 
# @desc :

TASK = "advertisement"
EXPORT_TOP_K = 100

local_data = './data'
lock_file_1 = "running_1.lock"  # python程序正在执行的 锁文件
lock_file_2 = "running_2.lock"  # python程序正在执行的 锁文件
lock_file_2_1 = "running_2_1.lock"  # python程序正在执行的 锁文件
lock_file_2_2 = "running_2_2.lock"  # python程序正在执行的 锁文件
lock_file_2_3 = "running_2_3.lock"  # python程序正在执行的 锁文件
lock_file_3 = "running_3.lock"  # python程序正在执行的 锁文件
lock_file_4 = "running_4.lock"  # python程序正在执行的 锁文件

collection_base_xlnet_768 = 'base_xlnet768'  # milvus库名
collection_name_base_768 = 'bert768_base'  # milvus库名
collection_name_base_1024 = 'bert1024_base'  # milvus库名
collection_name_base_2048 = 'resnet2048_base'  # milvus库名

# 李鑫=================================
TITLE_ADD = '_TITLE_ADD'
TITLE_FULL = '_TITLE_FULL'
TITLE_FULL_1 = '_TITLE_FULL_1'

hdfs_host = 'hdfs://127.0.0.1:9870'  # hdfs远程数据源
hdfs_hosts = '127.0.0.1:9870,127.0.0.1:9870'  # hdfs远程数据源服务器
# hdfs数据源配置
HDFS_TITLE_FULL_DIR = 'hdfs://mesos1:9870/data/recomander/bert_milvus/itemcode_full'
HDFS_TITLE_FULL_DIR_1 = 'hdfs://mesos1:9870/data/similar/bert_base'
HDFS_TILE_ADD_DIR = 'hdfs://mesos1:9870/data/recomanr/bert_milvus/itemcode_add'

LOCAL_TITLE_FULL_DIR = './data/title_full'
LOCAL_TITLE_FULL_DIR_1 = './data/title_full_1'
LOCAL_TITLE_ADD_DIR = './data/title_add'
LOCAL_VECTOR_ADD_DIR = './data/vector_add'
LOCAL_VECTOR_FULL_DIR = './data/vector_full'
LOCAL_VECTOR_FULL_DIR_1 = './data/vector_full_1'
LOCAL_EXPORT_DIR = './data/export'
LOCAL_EXPORT_DIR_1 = './data/export_1'

outdir_bert = "hdfs://mesos1:9870/data/recomander/bert"
outdir_bert_1 = "hdfs://mesos1:9870/data/similar/bert_res"

# =============bert itemnew====================
TITLE_ITEMNEW = '_TITLE_ITEMNEW'
PRICE = '_PRICE'
###针对itemnew的配置
HDFS_TITLE_DIR_itemnew = 'hdfs://mesos1:9870/data/recomander/bert_milvus/itemcode_new'

LOCAL_VECTOR_DIR_itemnew = './data/vector_itemnew'
LOCAL_TITLE_DIR_itemnew = './data/title_itemnew'
LOCAL_EXPORT_DIR_itemnew = './data/export_itemnew'
LOCAL_EXPORT_DIR_itemnew_2 = '/data/apps/recall-offline/15Rec/export_itemnew'

collection_name_recom = 'xlnet768_recom'
param_li = {'collection_name': collection_name_recom, 'dimension': 768, 'index_file_size': 1024}  # milvus库参数

outdir_ods_simi_prod = "hdfs://nameservice1/dw/ods/ods_simi_prod"  # nameservice1
outdir_ods_simi_itemnew = "hdfs://nameservice1/dw/ods/ods_simi_itemnew"
# 生产
local_ods_simi_prod = '/data/apps/recall-offline/data/export_itemnew/ods_simi_prod'
hdfs_ods_simi_prod = '  hdfs://nameservice1/dw/ods/ods_simi_prod/'
local_ods_simi_itemnew = '/data/apps/recall-offline/data/export_itemnew/ods_simi_itemnew'
hdfs_ods_simi_itemnew = '  hdfs://nameservice1/dw/ods/ods_simi_itemnew/'

'''
=============bert crocess store   跨店满减====================
'''
TITLE_CROCESS_STORE = '_TITLE_CROCESS_STORE'  # 任务标签
HDFS_TITLE_DIR_crocess_store = 'hdfs://127.0.0.1:9870/data/cross_store_recall/base_data'  # 源数据
LOCAL_TITLE_DIR_crocess_store = './data/title_crocess_store'  # 源数据 本地存储
LOCAL_VECTOR_DIR_crocess_store = './data/vector_crocess_store'  # 向量化数据
LOCAL_EXPORT_DIR_crocess_store = './data/export_crocess_store'  # 计算相似度后数据
collection_name_crocess_store = 'bert128_crocess_store'  # milvus库名
param_crocess_store = {'collection_name': collection_name_crocess_store, 'dimension': 768,'index_file_size': 1024}  # milvus库参数
outdir_crocess_store = "hdfs://127.0.0.1:9870/data/cross_store_recall/res_data"  # 输出文件位置

'''
=============bert pc_home_page  pc 首页优化 绿区批发====================
'''
TITLE_GREEN_WHOLSALE = '_TITLE_GREEN_WHOLSALE'  # 任务标签
HDFS_TITLE_DIR_GREEN_WHOLSALE = 'hdfs://127.0.0.1:9870/data/pc_home_page/bert_base'  # 源数据
LOCAL_TITLE_DIR_GREEN_WHOLSALE = './data/title_green_wholesale'  # 源数据 本地存储
LOCAL_VECTOR_DIR_GREEN_WHOLSALE = './data/vector_green_wholesale'  # 向量化数据
LOCAL_EXPORT_DIR_GREEN_WHOLSALE = './data/export_green_wholesale'  # 计算相似度后数据
collection_name_GREEN_WHOLSALE = 'green_wholesale'  # milvus库名
param_GREEN_WHOLSALE = {'collection_name': collection_name_GREEN_WHOLSALE, 'dimension': 768, 'index_file_size': 1024}  # milvus库参数
outdir_GREEN_WHOLSALE = "hdfs://127.0.0.1:9870/data/pc_home_page/bert_res"  # 输出文件位置

'''
=============bert advert   站外广告====================
'''
TITLE_ADVERT = '_TITLE_ADVERT'  # 任务标签
HDFS_TITLE_DIR_advert = 'hdfs://127.0.0.1:9870/data/adrec_recall/base_data'  # 源数据
LOCAL_TITLE_DIR_advert = './data/title_advert'  # 源数据 本地存储
LOCAL_VECTOR_DIR_advert = './data/vector_advert'  # 向量化数据
LOCAL_EXPORT_DIR_advert = './data/export_advert'  # 计算相似度后数据
collection_name_advert = 'advert'  # milvus库名
param_advert = {'collection_name': collection_name_advert, 'dimension': 768, 'index_file_size': 1024}  # milvus库参数
outdir_advert = "hdfs://127.0.0.1:9870/data/adrec_recall/res_data"  # 输出文件位置

'''
=============item2vec ====================
'''
VECTOR_ITEM2VEC = '_ITEM2VEC'  # 任务标签
HDFS_VECTOR_DIR_item2vec = 'hdfs://127.0.0.1:9870/data/item2vec'  # 源数据
LOCAL_VECTOR_DIR_item2vec = './data/vector_item2vec'  # 向量化数据
LOCAL_EXPORT_DIR_item2vec = './data/export_item2vec'  # 计算相似度后数据
collection_name_item2vec = 'item2vec'  # milvus库名
param_item2vec = {'collection_name': collection_name_item2vec, 'dimension': 100, 'index_file_size': 1024}  # milvus库参数
outdir_item2vec = "hdfs://127.0.0.1:9870/data/exportItem2vec"  # 输出文件位置

'''
=============item2vec 128纬度====================
'''
VECTOR_ITEM2VEC100 = '_ITEM2VEC100'  # 任务标签
HDFS_VECTOR_DIR_item2vec100 = 'hdfs://127.0.0.1:9870/data/item2vec_home'  # 源数据
LOCAL_VECTOR_DIR_item2vec100 = './data/vector_item2vec128'  # 向量化数据
LOCAL_EXPORT_DIR_item2vec100 = './data/export_item2vec128'  # 计算相似度后数据
collection_name_item2vec100 = 'item2vec100'  # milvus库名
param_item2vec100 = {'collection_name': collection_name_item2vec100, 'dimension': 100, 'index_file_size': 1024}  # milvus库参数
outdir_item2vec100 = "hdfs://127.0.0.1:9870/data/export_item2vec_home"  # 输出文件位置

'''
=============item2vec user 100纬度====================
'''
VECTOR_ITEM2VEC_USER = '_ITEM2VEC_USER'  # 任务标签
HDFS_VECTOR_DIR_item2vec_user = 'hdfs://127.0.0.1:9870/data/user2vec_jfy_home/res'  # 源数据
LOCAL_VECTOR_DIR_item2vec_user = './data/vector_item2vec_user'  # 向量化数据
LOCAL_EXPORT_DIR_item2vec_user = './data/export_item2vec_user'  # 计算相似度后数据
collection_name_item2vec_user = 'user2vec'  # milvus库名
param_item2vec_user = {'collection_name': collection_name_item2vec_user, 'dimension': 100, 'index_file_size': 1024}  # milvus库参数
outdir_item2vec_user = "hdfs://127.0.0.1:9870/data/user2vec_jfy_home/export_res_u_search_u"  # 输出文件位置

'''
=============item2vec user2item ====================
'''
VECTOR_ITEM2VEC_U2I = '_ITEM2VEC_U2I'  # 任务标签
HDFS_VECTOR_DIR_item2vec_u2i = 'hdfs://127.0.0.1:9870/data/user2vec_jfy_home/res'  # 源数据
LOCAL_VECTOR_DIR_item2vec_u2i = './data/vector_item2vec_u2i'  # 向量化数据
LOCAL_EXPORT_DIR_item2vec_u2i = './data/export_item2vec_u2i'  # 计算相似度后数据
collection_name_item2vec_u2i = 'item2vec'  # milvus库名
outdir_item2vec_u2i = "hdfs://127.0.0.1:9870/data/user2vec_jfy_home/export_res_u_search_i"  # 输出文件位置
'''
=============  graph  ====================
'''
VECTOR_GRAPH = '_GRAPH'  # 任务标签
HDFS_VECTOR_DIR_graph = 'hdfs://127.0.0.1:9870/data/graphembedding'  # 源数据
LOCAL_VECTOR_DIR_graph = './data/vector_graph'  # 向量化数据
LOCAL_EXPORT_DIR_graph = './data/export_graph'  # 计算相似度后数据
collection_name_graph = 'graph'  # milvus库名
param_graph = {'collection_name': collection_name_graph, 'dimension': 100, 'index_file_size': 1024}  # milvus库参数
outdir_graph = "hdfs://127.0.0.1:9870/data/exportGraphembedding"  # 输出文件位置

'''
============dssml item2item ====================
'''
VECTOR_DSSMLI2I = '_DSSMLI2I'  # 任务标签
HDFS_VECTOR_DIR_dssmli2i = 'hdfs://127.0.0.1:9870/data/dssmI2I'  # 源数据
LOCAL_VECTOR_DIR_dssmli2i = './data/vector_dssmli2i'  # 向量化数据
LOCAL_EXPORT_DIR_dssmli2i = './data/export_dssmli2i'  # 计算相似度后数据
collection_name_dssmli2i = 'linliping_items'  # milvus库名
outdir_dssmli2i = "hdfs://127.0.0.1:9870/data/exportDssmI2I"  # 输出文件位置

'''
=============dssml user2item ====================
'''
VECTOR_DSSMLU2I = '_DSSMLU2I'  # 任务标签
HDFS_VECTOR_DIR_dssmlu2i = 'hdfs://127.0.0.1:9870/data/dssmU2I'  # 源数据
LOCAL_VECTOR_DIR_dssmlu2i = './data/vector_dssmlu2i'  # 向量化数据
LOCAL_EXPORT_DIR_dssmlu2i = './data/export_dssmlu2i'  # 计算相似度后数据
collection_name_dssmlu2i = 'linliping_items'  # milvus库名
outdir_dssmlu2i = "hdfs:/127.0.0.1:9870/data/exportDssmU2I"  # 输出文件位置

'''
============= 广告  1h/更新 ====================
'''
TARGET_ADVERT_ONLINE = '_ADVERT_ONLINE'  # 任务标签
TARGET_ADVERT_ADD = '_ADVERT_ADD'  # 任务标签
TARGET_ADVERT_ONLINE_IMAGE = '_ADVERT_ONLINE_IMAGE'  # 任务标签
TARGET_ADVERT_BASE = '_ADVERT_BASE'  # 任务标签
HDFS_TITLE_DIR_ADVERT_ONLINE = 'hdfs://127.0.0.1:9870/data/ad_on_site/online'  # 源数据——广告在线数据（1小时更新）
HDFS_TITLE_DIR_ADVERT_ADD = 'hdfs://127.0.0.1:9870/data/ad_on_site/add_cilck'  # 源数据——广告当天增量数据
HDFS_TITLE_DIR_ADVERT_BASE = 'hdfs://127.0.0.1:9870/data/ad_on_site/base_cilck'  # 源数据——主商品
LOCAL_TITLE_DIR_ADVERT_ONLINE = './data/title_advert_online'  # 本地源数据——广告在线数据（1小时更新）
LOCAL_TITLE_DIR_ADVERT_ADD = './data/title_advert_add'  # 本地源数据——广告当天增量数据
LOCAL_TITLE_DIR_ADVERT_BASE = './data/title_advert_base'  # 本地源数据——主商品
LOCAL_VECTOR_DIR_ADVERT_ONLINE = './data/vector_advert_online'  # 向量化数据——广告在线数据（1小时更新）（废弃）
LOCAL_VECTOR_DIR_ADVERT_ADD = './data/vector_advert_add'  # 向量化数据
LOCAL_VECTOR_DIR_ADVERT_BASE = './data/vector_advert_base'  # 向量化数据
LOCAL_EXPORT_DIR_ADVERT_ONLINE = './data/export_advert_online'  # 计算相似度后数据
LOCAL_EXPORT_DIR_ADVERT_ONLINE_IMAGE = './data/export_advert_online_image'
collection_name_ADVERT_ONLINE = 'advert_online'  # milvus库名
collection_name_ADVERT_ONLINE_IMAGE = 'advert_online_image'  # milvus库名
param_ADVERT_ONLINE = {'collection_name': collection_name_ADVERT_ONLINE, 'dimension': 768, 'index_file_size': 1024}  # milvus库参数
param_ADVERT_ONLINE_IMAGE = {'collection_name': collection_name_ADVERT_ONLINE_IMAGE, 'dimension': 2048, 'index_file_size': 1024}  # milvus库参数
OUT_DIR__ADVERT_ONLINE = "hdfs://127.0.0.1:9870/majiashu/ad_on_site/export_title_online"  # 输出文件位置

'''
=============标签 item2item ====================
'''
TARGET_EMB_I2I = '_EMB_I2I'  # 任务标签
HDFS_VECTOR_DIR_emb_item = 'hdfs://127.0.0.1:9870/data/emb_v2/item'  # 源数据
LOCAL_VECTOR_DIR_emb_item = './data/vector_emb_item'  # 向量化数据
LOCAL_EXPORT_DIR_emb_i2i = './data/export_emb_i2i'  # 计算相似度后数据
collection_name_emb_item64 = 'emb_item64'  # milvus库名
param_EMB_I2I = {'collection_name': collection_name_emb_item64, 'dimension': 64, 'index_file_size': 1024}  # milvus库参数
outdir_item2item_ver = "hdfs://127.0.0.1:9870/data/export_item2item_ver"  # 输出文件位置
'''
============标签 user2user ====================
'''
TARGET_EMB_U2U = '_EMB_U2U'  # 任务标签
HDFS_VECTOR_DIR_emb_user = 'hdfs://127.0.0.1:9870/data/emb_v2/user'  # 源数据
LOCAL_VECTOR_DIR_emb_user = './data/vector_emb_user'  # 向量化数据
LOCAL_EXPORT_DIR_emb_u2u = './data/export_emb_u2u'  # 计算相似度后数据
collection_name_emb_user64 = 'emb_user64'  # milvus库名
param_EMB_U2U = {'collection_name': collection_name_emb_user64, 'dimension': 64, 'index_file_size': 1024}  # milvus库参数
outdir_user2user_ver = "hdfs://127.0.0.1:9870/data/export_user2user_ver"  # 输出文件位置
'''
=============标签 user2item ====================
'''
TARGET_EMB_U2I = '_EMB_U2I'  # 任务标签
LOCAL_EXPORT_DIR_emb_u2i = './data/export_emb_u2i'  # 计算相似度后数据
outdir_user2item_ver = "hdfs://127.0.0.1:9870/data/export_user2item_ver"  # 输出文件位置

'''
============= word2vec ====================
'''
TARGET_EMB_WORD2VEC = '_WORD2VEC'  # 任务标签
HDFS_VECTOR_DIR_word2vec = 'hdfs://127.0.0.1:9870/data/word2vecFromTorch'  # 源数据
LOCAL_VECTOR_DIR_word2vec = './data/vector_word2vec'  # 向量化数据
LOCAL_EXPORT_DIR_word2vec = './data/export_word2vec'  # 计算相似度后数据
collection_name_word2vec = 'word2vec'  # milvus库名
param_word2vec = {'collection_name': collection_name_word2vec, 'dimension': 128, 'index_file_size': 1024}  # milvus库参数
outdir_word2vec = "hdfs://127.0.0.1:9870/data/export_word2vecFromTorch"  # 输出文件位置

'''
=============deepwalk =======================
'''
TARGET_EMB_DEEPWALK = '_DEEPWALK'  # 任务标签
HDFS_VECTOR_DIR_deepwalk = 'hdfs://127.0.0.1:9870/data/deepwalkword2vecFromTorch'  # 源数据
LOCAL_VECTOR_DIR_deepwalk = './data/vector_deepwalk'  # 向量化数据
LOCAL_EXPORT_DIR_deepwalk = './data/export_deepwalk'  # 计算相似度后数据
collection_name_deepwalk = 'deepwalk'  # milvus库名
param_deepwalk = {'collection_name': collection_name_deepwalk, 'dimension': 128, 'index_file_size': 1024}  # milvus库参数
outdir_deepwalk = "hdfs://127.0.0.1:9870/data/export_deepwalkFromTorch"  # 输出文件位置
