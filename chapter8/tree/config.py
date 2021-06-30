#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @project : books_method
# @File : config.py
# @Time    : 2020/7/6 14:25
# @Author  : Michael.liu

params = {
    "task": "train",
    "boosting": "gbdt",
    "objective": "binary",
    "tree_learner": "serial",
    "metric": ['auc','binary_logloss'],
    "training_metric": True,
    #'.\\..\\..\\..\\data\\model_data.txt'
    "train_data": 'train.csv',
    "test_data":  'test.csv',
    "header": "true",
    "label_column": "name:target",
    #"weight_column": "name:weight",
    "ignore_column" : "name:applicantFirst",
    "categorical_feature": '',
    #	covered_query_term_number=apply_flag
    #"name:pubid	name:fieldsName	name:fieldWords	name:techWords	name:funcWords	name:tfidf_v1	name:goodsList	name:warnlevelRe	name:indLen	name:claimsIndCount	name:feature	name:apply_flag",
    #"pubid				fieldsName				fieldWords				techWords				funcWords				tfidf_v1				goodsList				warnlevelRe				indLen				claimsIndCount				feature				apply_flag",
    #"name:covered_query_term_number=body,covered_query_term_number=anchor,covered_query_term_number=title,covered_query_term_number=url,covered_query_term_number=whole_document" ,
    "metric_freq": 1,
    "max_bin": 255,
    "num_trees": 100,
    "learning_rate": 0.225,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "min_data_in_leaf": 100,
    "min_sum_hessian_in_leaf": 5,
    "num_threads": 12,
    "is_sparse": True,
    "two_round": False,
    "convert_model_language": "cpp",
    "output_model": "model.md",
    "output_result":"gbm_pre.txt"
    }

