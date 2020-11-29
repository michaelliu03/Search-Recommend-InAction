#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @project : books_method
# @File : feature_kind.py
# @Time    : 2020/6/24 15:43
# @Author  : Zhaohy



head = 'id	query_cqtn_1_cat	query_cqtn_2_cat	query_cqtn_3_cat	query_cqtn_4_cat	query_cqtn_5_cat	query_sotf_1	query_sotf_2	query_sotf_3	query_sotf_4	query_sotf_5	query_motf_1	query_motf_2	query_motf_3	query_motf_4	query_motf_5	query_motf_1	query_motf_2	query_motf_3	query_motf_4	query_motf_5	query_motf_1	query_motf_2	query_motf_3	query_motf_4	query_motf_5	query_votf_1	query_votf_2	query_votf_3	query_votf_4	query_votf_5	query_bm_1_bin	query_bm_2_bin	query_bm_3_bin	query_bm_4_bin	query_bm_5_bin	query_other_1	query_other_1'


cats = []

bins = []

numbers = []
for i in head.split('\t'):

    if 'cat' in i :
        cats.append(i)
    if 'bin' in i :
        bins.append(i)
    else:
        numbers.append(i)


print(cats)

print(bins)

print(numbers)

