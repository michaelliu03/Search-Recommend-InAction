#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:helper.py
# @Author: Michael.liu
# @Date:2020/6/3 14:05
# @Desc: this code is ....
import json


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit






tr_csv_all_path = 'avazu_ctr/train.csv'
tr_csv_path = 'avazu_ctr/train_sample.csv'
ts_csv_path = 'avazu_ctr/test.csv'
data_type = {'id': 'U', 'hour': 'U', 'device_type': 'U', 'C1': 'U', 'C15': 'U', 'C16': 'U'}


# 分层采样数据
def genSampleData():
    train_all = pd.read_csv(tr_csv_all_path)
    split_data = StratifiedShuffleSplit(n_splits=1, train_size=0.05, random_state=42)
    for train_index, test_index in split_data.split(train_all, train_all["click"]):
        strat_train_set = train_all.loc[train_index]
        strat_train_set.to_csv("avazu_ctr/train_sample.csv", header=True)
    # 特征工程


def futureEngineer():
    print("step 2-- future engineer")
    train = pd.read_csv(tr_csv_path, dtype=data_type, index_col='id')
    test = pd.read_csv(ts_csv_path, dtype=data_type, index_col='id')
    test.insert(0, 'click', 0)
    tr_ts = pd.concat([test, train], copy=False)

    # site_id
    site_id_count = tr_ts.site_id.value_counts()
    site_id_category = {}
    site_id_category[0] = site_id_count.loc[site_id_count > 20].index.values
    site_id_category[1] = site_id_count.loc[site_id_count <= 20].index.values

    site_id_C_type_dict = {}
    for key, values in site_id_category.items():
        for item in values:
            site_id_C_type_dict[str(item)] = key

    json.dump(site_id_C_type_dict, open("site_id_C_type_dict.json", "w"))

    # site_domain
    site_domain_count = tr_ts.site_domain.value_counts()
    site_domain_category = {}
    site_domain_category[0] = site_domain_count.loc[site_domain_count > 20].index.values
    site_domain_category[1] = site_domain_count.loc[site_domain_count <= 20].index.values

    site_domain_C_type_dict = {}
    for key, values in site_domain_category.items():
        for item in values:
            site_domain_C_type_dict[str(item)] = key

    json.dump(site_domain_C_type_dict, open("site_domain_C_type_dict.json", "w"))

    # app_id
    app_id_count = tr_ts.app_id.value_counts()
    app_id_category = {}
    app_id_category[0] = app_id_count.loc[app_id_count > 20].index.values
    app_id_category[1] = app_id_count.loc[app_id_count <= 20].index.values

    app_id_C_type_dict = {}
    for key, values in app_id_category.items():
        for item in values:
            app_id_C_type_dict[str(item)] = key

    json.dump(app_id_C_type_dict, open("app_id_C_type_dict.json", "w"))

    # device_model
    device_model_count = tr_ts.device_model.value_counts()
    device_model_category = {}
    device_model_category[0] = device_model_count.loc[device_model_count > 200].index.values
    device_model_category[1] = device_model_count.loc[device_model_count <= 200].index.values

    device_model_C_type_dict = {}
    for key, values in device_model_category.items():
        for item in values:
            device_model_C_type_dict[str(item)] = key

    json.dump(device_model_C_type_dict, open("device_model_C_type_dict.json", "w"))


def train_test_split():
    train = pd.read_csv(tr_csv_path, dtype=data_type, index_col='id')
    test = pd.read_csv(ts_csv_path, dtype=data_type, index_col='id')
    test.insert(0, 'click', 0)

    tr_ts = pd.concat([test, train], copy=False)

    tr_ts['day'] = tr_ts['hour'].apply(lambda x: x[-4:-2])
    tr_ts['hour'] = tr_ts['hour'].apply(lambda x: x[-2:])

    tr_ts['is_device'] = tr_ts['device_id'].apply(lambda x: 0 if x == 'a99f214a' else 1)  # 详见探索性数据分析部分

    app_id_C_type_dict = json.load(open("app_id_C_type_dict.json", "r"))
    site_id_C_type_dict = json.load(open("site_id_C_type_dict.json", "r"))
    site_domain_C_type_dict = json.load(open("site_domain_C_type_dict.json", "r"))
    device_model_C_type_dict = json.load(open("device_model_C_type_dict.json", "r"))

    tr_ts['C_app_id'] = tr_ts["app_id"].apply(lambda x: x if app_id_C_type_dict.get(x) == 0 else "other_app_id")
    tr_ts['C_site_id'] = tr_ts['site_id'].apply(lambda x: x if site_id_C_type_dict.get(x) == 0 else "other_site_id")
    tr_ts['C_site_domain'] = tr_ts['site_domain'].apply(
        lambda x: x if site_domain_C_type_dict.get(x) == 0 else "other_site_domain")
    tr_ts['C_device_model'] = tr_ts['device_model'].apply(
        lambda x: x if device_model_C_type_dict.get(x) == 0 else "other_device_model")

    tr_ts["C_pix"] = tr_ts["C15"] + '&' + tr_ts["C16"]
    tr_ts["C_device_type_1"] = tr_ts["device_type"] + '&' + tr_ts["C1"]

    tr_ts.drop(
        ['device_id', "device_type", 'app_id', 'site_id', 'site_domain', 'device_model', "C1", "C17", 'C15', 'C16'],
        axis=1, inplace=True)

    lenc = preprocessing.LabelEncoder()
    C_fields = ['hour', 'banner_pos', 'site_category', 'app_domain', 'app_category',
                'device_conn_type', 'C14', 'C18', 'C19', 'C20', 'C21', 'is_device', 'C_app_id', 'C_site_id',
                'C_site_domain', 'C_device_model', 'C_pix', 'C_device_type_1']
    for f, column in enumerate(C_fields):
        print("convert " + column + "...")
        tr_ts[column] = lenc.fit_transform(tr_ts[column])

    dummies_site_category = pd.get_dummies(tr_ts['site_category'], prefix='site_category')
    dummies_app_category = pd.get_dummies(tr_ts['app_category'], prefix='app_category')

    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(tr_ts[['C14', 'C18', 'C19', 'C20', 'C21']])
    tr_ts[['C14', 'C18', 'C19', 'C20', 'C21']] = age_scale_param.transform(tr_ts[['C14', 'C18', 'C19', 'C20', 'C21']])

    tr_ts_new = pd.concat([tr_ts, dummies_site_category, dummies_app_category], axis=1)
    tr_ts_new.drop(['site_category', 'app_category'], axis=1, inplace=True)

    tr_ts_new.iloc[:test.shape[0], ].to_csv('test_FE.csv')
    tr_ts_new.iloc[test.shape[0]:, ].to_csv('train_FE.csv')


if __name__ == '__main__':
    print(">>>>>start...")
    # genSampleData()
    # futureEngineer()
    train_test_split()
    print("finished!")