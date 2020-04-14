#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:Segment.py
# @Author: Michael.liu
# @Date:2019/2/12
# @Desc: NLP Segmentation ToolKit - Hanlp Python Version


def is_chinese(s):
    for ch in s:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def is_normal(self, s):
    s = s.replace('\r', '').replace('\n', '').replace('\t', '')
    return s






