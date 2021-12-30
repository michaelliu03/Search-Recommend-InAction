#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 
# @desc
from __future__ import unicode_literals
import argparse
import datetime
import os

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession

INDEX = None
VALID_INDEX = ("chollima_resume", "chollima_resume_cache")
is_online = False


def formatter(row):
    data = {}

    def now_title_add_one_three(res_title, rw_index, rw_title):
        if not res_title:
            res_title = ""
        if not rw_index or not rw_title:
            return res_title, res_title

        sorted_title = [title for key, title in sorted(zip(rw_index, rw_title), key=lambda x: x[0])]
        one_title = [sorted_title[0]]
        if len(sorted_title) == 1:
            three_title = one_title
        else:
            three_title = sorted_title[:3]
        one_title.insert(0, res_title)
        three_title.insert(0, res_title)
        now_add_one_title_result = list(set(one_title))
        now_add_three_title_result = list(set(three_title))
        return " ".join(now_add_one_title_result), " ".join(now_add_three_title_result)

    def set_data(key, value):
        """
        对字段是否为空判定并赋值
        :param key:
        :param value:
        :return:
        """
        if value is None or value == "NULL":
            data[key] = ' '
        else:
            data[key] = value

    if row.rw_title:
        title_str = "".join(row.rw_title)
        title_str = title_str.replace('—', '')
        title_str = title_str.replace('…', '')
        title_str = title_str.replace('-', '')
        if title_str is not None and title_str != '':
            now_add_one, now_add_three = now_title_add_one_three(row.res_title, row.rw_index, row.rw_title)
            set_data("work_exp_title_his", " ".join(row.rw_title))
            set_data("work_exp_title_now_one", now_add_one)
            set_data("work_exp_title_now_three", now_add_three)

    set_data("work_exp_duty", row.work_exp_duty)
    # 项目经历中担任职务rpd_title和项目职责rpd_duty的拼接在一起的字段project_exp_duty
    set_data("project_exp_duty", row.project_exp_duty)
    now = datetime.datetime.now().strftime(u"%Y-%m-%d %H:%M:%S")
    set_data("update_time", now)
    return data

def process(partition):
    # from eswrite import ESClient
    # from eswrite import ESCluster
    # from eswrite import ESDocAsUpsert
    from chapter11 import search
    data = list(partition)


    es_client = search.Search(is_online=is_online,
                              hosts=[{"host": "192.168.64.44", "port": 5811},
                                     {"host": "192.168.64.45", "port": 5811},
                                     {"host": "192.168.64.46", "port": 5811}])
    result = []
    chunk_size = 400

    for row in data:
        data_json = formatter(row)
        if data_json is None:
            continue
        data_json = es_client.get_json(index=INDEX, doc_type="_doc", field_name="doc",
                                       _id=row.res_id, data=data_json, action="update")
        if data_json is None:
            continue
        result.append(data_json)

        if len(result) == chunk_size:
            print("start to write %d data to es" % len(result))
            es_errors = es_client.bulk(result)
            result = []
            print(es_errors)

    if result:
        print("start to write %d data to es" % len(result))
        errors = es_client.bulk(result)
        print(errors)
    print("write data to es end")
    yield len(data)

def init_sql(args, spark_context):
    if args.is_whole:
        sql_file = os.path.join("/spark/data/bole", "user_resume_skill_word_whole.sql")
        sql = "\n".join(spark_context.textFile(sql_file).collect())
    else:
        sql_file = os.path.join("/spark/data/bole", "user_resume_skill_word.sql")
        sql = "\n".join(spark_context.textFile(sql_file).collect())
    try:
        print(sql.encode("gbk"))
    except:
        print(sql.encode("utf8"))
    return sql

if __name__ == "__main__":
    parser = argparse.ArgumentParser("process skill word")
    parser.add_argument("--is-whole", dest="is_whole", help="update whole data or increment.",
                        action="store_true")
    parser.add_argument("is_online", help="索引online or offline", default="offline")
    parser.add_argument("-i", "--index_type", dest="index_type", default="bole_resume",
                        help="写的索引类型, 有效值: %s" % ",".join(VALID_INDEX))
    args = parser.parse_args()

    if args.index_type not in VALID_INDEX:
        raise ValueError("请输入有效的索引 %s" % ",".join(VALID_INDEX))
    INDEX = args.index_type
    if args.is_online == "online":
        is_online = True
        print("online index")
    if args.is_online == "offline":
        is_online = False
        print("offline index")
    spark_conf = SparkConf().setAppName("bole_skill_word")
    spark_conf.setMaster("yarn-cluster")
    spark_context = SparkContext(conf=spark_conf)
    spark = SparkSession.builder.config(conf=spark_conf).enableHiveSupport().getOrCreate()
    sql = init_sql(args, spark_context)
    df = spark.sql(sql)
    summer = df.rdd.repartition(20).mapPartitions(process).sum()
    print("update resume skill word fail number: %d " % summer)