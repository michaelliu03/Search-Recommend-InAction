#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:__init__.py.py
# @Author: Michael.liu
# @Date:2020/6/19 19:57
# @Desc: this code is ....

#在讲解召回策略时介绍了协同过滤，这里给了一个协同过滤的例子
# 整个例子在spark上执行通过，采用了天池数据进行讲解，数据具体可以参考：
# https://tianchi.aliyun.com/dataset/dataDetail?dataId=649

#本数据集包含了2017年11月25日至2017年12月3日之间，有行为的约一百万随机用户的所有行为（行为包括点击、购买、加购、喜欢）。数据集的组织形式和MovieLens-20M类似，即数据集的每一行表示一条用户行为，由用户ID、商品ID、商品类目ID、行为类型和时间戳组成，并以逗号分隔。关于数据集中每一列的详细描述如下：
#列名称	说明
#用户ID	整数类型，序列化后的用户ID
#商品ID	整数类型，序列化后的商品ID
#商品类目ID	整数类型，序列化后的商品所属类目ID
#行为类型	字符串，枚举类型，包括('pv', 'buy', 'cart', 'fav')
#时间戳	行为发生的时间戳

#购买行为一般由下面4个动作构成 浏览，加购物车，收藏，购买
#购买是最终状态，加购物车，浏览详情，收藏，按照这个顺序依次递减
# buy: 5' 购买是最终的状态
# cart ：3‘ 加购物车
# pv：2’ 浏览详情
# fav：1‘ 收藏

# 因为这个数据集还是比较大的，开始处理起来有点费劲，最后考虑后采用了pyspark，
# 关于spark的一些理论在书中我并没有介绍，但是在离线处理过程中Big Data
# 一些处理还是需要采用的