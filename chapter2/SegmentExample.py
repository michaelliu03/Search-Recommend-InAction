#!/usr/bin/env python
# -*-coding:utf-8-*-
# @File:SegmentExample.py
# @Author: Michael.liu
# @Date:2019/2/12
# @Desc: 这个文档测试了hanlp分词器的应用效果

from pyhanlp import *
import os

testCases = [
    "商品和服务",
    "结婚的和尚未结婚的确实在干扰分词啊",
    "买水果然后来世博园最后去世博会",
    "中国的首都是北京",
    "欢迎新老师生前来就餐",
    "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
    "随着页游兴起到现在的页游繁盛，依赖于存档进行逻辑判断的设计减少了，但这块也不能完全忽略掉。"]


def pyHanlpSeg():
    for sentence in testCases:
        seg_list = HanLP.segment("".join(sentence))
        hanlpSegList = []
        for item in seg_list:
            hanlpSegList.append(item.word)
        print(hanlpSegList)


def pyHanlpSeg(sentence):
    seg_list = HanLP.segment(sentence)
    hanlpSegList = []
    for item in seg_list:
        hanlpSegList.append(item.word)
    return hanlpSegList


if __name__ == '__main__':
    print("Hanlp Seg")
    pyHanlpSeg()

# jpype.shutdownJVM()
