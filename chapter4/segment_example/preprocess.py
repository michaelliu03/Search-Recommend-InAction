#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:preprocess.py
# @Author: Michael.liu
# @Date:2020/5/8 17:31
# @Desc: this code is ....
import codecs
import os

def character_tagging(input_file_, output_file_):
    input_data = codecs.open(input_file_, 'r', 'utf-8')
    output_data = codecs.open(output_file_, 'w', 'utf-8')
    for line in input_data.readlines():
        # 移除字符串的头和尾的空格。
        word_list = line.strip().split()
        for word in word_list:
            words = word.split("/")
            word = words[0]
            if len(word) == 1:
                if word == '。' or word == '？' or word == '！':
                    output_data.write("\n")
                else:
                    output_data.write(word + "/S ")
            elif len(word) >= 2:
                output_data.write(word[0] + "/B ")
                for w in word[1: len(word) - 1]:
                    output_data.write(w + "/M ")
                output_data.write(word[len(word) - 1] + "/E ")
        output_data.write("\n")
    input_data.close()
    output_data.close()

if __name__=="__main__":
    print("......start......")
    fileName1 = '../../data/chapter4/seg/msr_training.utf8'
    fileName2 = '../../data/chapter4/seg/pku_training.utf8'

    character_tagging(fileName1, "msr_training_out.csv")
    character_tagging(fileName2, "pku_training_out.csv")
    print(".....finished.....")