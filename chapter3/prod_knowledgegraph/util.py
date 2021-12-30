#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 
# @desc :


def clean(keys):
    res = []
    for key in keys.split('-'):
        ga = key.split('%')
        if len(ga) > 2: continue
        if len(ga) == 2:
            if (not ga[0].isdigit()) or (len(ga[1]) > 0):
                continue
        res.append(key)
    return res