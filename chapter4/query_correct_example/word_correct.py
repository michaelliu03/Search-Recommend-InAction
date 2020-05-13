#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:word_correct.py
# @Author: Michael.liu
# @Date:2020/5/9 16:08
# @Desc: this code is ....

from  pypinyin import *
import codecs

class WordCorrect:
     def __init__(self):
         self.char_path = 'char.utf8'
         self.model_path = 'query_correct.model'
         self.charlist = [word.strip() for word in codecs.open(self.char_path,'r','utf-8') if word.strip()]
         self.pinyin_dict = self.load_model(self.model_path)


     def load_model(self, model_path):
         f = open(model_path, 'r',encoding='utf-8')
         a = f.read()
         word_dict = eval(a)
         f.close()
         return word_dict

     def edit1(self, word):
        n = len(word)
        return set([word[0:i] + word[i + 1:] for i in range(n)] +  # deletion
                   [word[0:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(n - 1)] +  # transposition
                   [word[0:i] + c + word[i + 1:] for i in range(n) for c in self.charlist] +  # alteration
                   [word[0:i] + c + word[i:] for i in range(n + 1) for c in self.charlist])  # insertion


if __name__ == "__main__":
    corrector = WordCorrect()
    word = '我门'
    word_pinyin = ','.join(lazy_pinyin(word))
    candiwords = corrector.edit1(word)
    print(candiwords)
    print(word_pinyin)
    print(corrector.pinyin_dict.get(word_pinyin, 'na'))