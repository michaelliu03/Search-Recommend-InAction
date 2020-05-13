#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:__init__.py.py
# @Author: Michael.liu
# @Date:2020/5/8 15:23
# @Desc: this code is the sample fo query_expansion
# query 扩展的实例比较多，实现方式可简单也可复杂。
# 比如可以通过同义词进行扩展，也可以通过 Apriori （关联关系）分析的方法进行挖掘
# 这里只给一个比较简单实现逻辑。这个实现逻辑是基于同义词扩展的方式。
# 获得同义词目前最简单和最好用的方式是word2vector，这里暂用该方式进行讲解。
