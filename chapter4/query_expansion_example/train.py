#!/usr/bin/env python
#-*-coding:utf-8-*-

import logging

from gensim.models import word2vec

corpus_file_path = "./jobtitle_title_JD_seg.txt"
def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence(corpus_file_path)
    model = word2vec.Word2Vec(sentences, size=250)
    #保存模型
    model.save("word2vec.model")


if __name__ == "__main__":
    main()