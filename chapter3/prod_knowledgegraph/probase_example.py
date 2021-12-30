#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : knowledge-graph-example
# @desc :这里需要请求一个服务

import re
import pandas as pd
import  requests
import  json
from chapter3.prod_knowledgegraph.util import clean
import math
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import sys

PROBASE_URL = "" #这里upload时需要去掉
CONNECTION_SIZE = 1000
session = requests.Session()
retry =Retry(connect = CONNECTION_SIZE,backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry, pool_maxsize=CONNECTION_SIZE, pool_connections=CONNECTION_SIZE)
session.mount('http://', adapter)
session.mount('https://', adapter)

def get_shiti(key_word,session):
    params = {"keyword": key_word,
              "openWhiteList": False, "platform": "pc",
              "requestType": "search", "size": 3,
              "vid": "rBIKGF6lR+5p7zgKD0K2Ag=5"}

    x = session.post(PROBASE_URL, data=json.dumps(params))  # 同一会话访问接口
    EntityResult = {'brandWords': [], 'categoryWords': [], 'attributeWords': []}
    try:
        EntityResult = json.loads(x.text)['entityResolutionResult']
    except:
        pass
    BrandWords = [brand['rawWord'] for brand in EntityResult['brandWords']]  # 品牌词
    CategoryWords = [category['rawWord'] for category in EntityResult['categoryWords']]   # 物品词
    Att_words = []  # 属性词
    OCC_words = []
    for att in EntityResult['attributeWords']:
        attrName = None
        try:
            attrName = att['attrName']
        except:
            continue
        if att['tag'] == "PER":
            continue
        if att['tag'] == "OCC":
            OCC_words.append(att['rawWord'])
            continue
        Att_words.append(att['rawWord'])
    return BrandWords, CategoryWords, Att_words, OCC_words


def process_data(file, file_w1, file_w2):
    top_attk = 4
    top_occk = 2
    i = 0
    clean_data = []
    cate_attr = dict()
    # f_out = open(file_w1, 'w', encoding="utf-8")

    for line in open(file, 'r', encoding='utf-8'):
        if i % 1e6 == 0: print('load data ', i)
        i += 1
        line = line.strip().split('@||@')

        if len(line) != 4: continue
        token = line[0]
        cate_token = line[1]
        title = line[2].lower()
        keywords = line[3].lower()
        keywords = clean(keywords)  # 清洗
        if len(keywords) < 2: continue
        brand, cate, attr, occ = get_shiti(title, session)
        _, k_cate, _, _ = get_shiti(" ".join(keywords), session)
        t_cates = cate[0] if len(cate) != 0 else "NULL"
        k_cates = k_cate[0] if len(k_cate) != 0 else "NULL"
        cates = k_cates if k_cates != "NULL" else t_cates
        # brands = '-'.join(brand) if len(brand) !=0 else "NULL"
        # print(cate)
        # [title, keyword]

        if k_cates == t_cates:
            if k_cates != "NULL":
                cate_attr = add_dcit(cate_attr, cates, attr)
        else:
            if k_cates != "NULL":
                cate_attr = add_dcit(cate_attr, k_cates, attr)
            if t_cates != "NULL":
                cate_attr = add_dcit(cate_attr, t_cates, attr)



    writer_cateattr(cate_attr, file_w2)
    print("Done!")

def add_dcit(cate_dict, cate, attr):
    if cate not in cate_dict.keys():
        cate_dict.setdefault(cate, dict())
    for att in attr:
        if len(att) < 2 and att.isalpha():
            continue
        if att not in cate_dict[cate].keys():
            cate_dict[cate].setdefault(att, 0)
        cate_dict[cate][att] += 1
    return cate_dict


def writer_cateattr(cate_dict, out_file):
    data = []
    # out_file = './amazon/cate_attr.csv'
    for cate, attr_dict in cate_dict.items():
        attr_dict = sorted(attr_dict.items(), key=lambda x: x[1], reverse=True)
        for attr in attr_dict:
            data.append([cate, attr[0], attr[1]])
    df = columns = ['cate', 'attr', 'freq']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(out_file, index=False)

if __name__ == '__main__':
    file  = "D://self-doc//self_project//Search-Recommend-InAction//data//chapter5//eval1"
    file_w1 = "amzon_test1.txt"
    file_w2 = "cate_attr.csv"
    process_data(file,file_w1,file_w2)