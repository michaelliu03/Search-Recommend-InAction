#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 
# @desc :
import os
import errno
import sentencepiece as spm
import re

from charpter5.transform.hparams import Hparams
import  logging
import random

logging.basicConfig(level=logging.INFO)

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

def prepro():
    clean_data = []
    i = 0
    fout = open('amazon/data2vocab.txt','w',encoding='utf-8')
    for line in open(hp.amazon_origin,'r',encoding='utf-8'):
        if i % 1e6 ==0: print('load data',i)
        i += 1
        line = line.strip().split('@||@')
        if len(line) != 4: continue
        title = line[2].lower()
        keywords = line[3].lower()
        keywords = clean(keywords)
        if len(keywords) < 2: continue
        clean_data.append([title,'-'.join(keywords)])  #[title, keywords]
        fout.write(title + '.' + '-'.join(keywords) + '\n')

    n = len(clean_data)

    print('all cleaned data = %d' % n)
    data_train1,data_train2,data_eva1,data_eva2,data_test1,data_test2 = [],[],[],[],[],[]
    fout1 = open('amazon/eval_o1.txt','w',encoding='utf-8')
    fout2 = open('amazon/eval_o2.txt','w',encoding='utf-8')

    logging.INFO("# Train a joint BPE model with sentencepiece")

    train = '--input=amazon/data2vocab.txt --pad_id=0 --unk_id=1 \
             --bos_id=2 --eos_id=3\
             --model_prefix = amazon/bpe --vocab_size={} \
             --model_type =bpe'.format(hp.vocab_size)

    logging.info("# Load trained bpe model")
    sp = spm.SentencePieceProcessor()

    sp.Load("amazon/bpe.model")

    def get_evaldata(sent1,sent2):
        piece1 = sp.EncodeAsPieces(sent1)
        piece2 = sp.EncodeAsPieces(sent2)
        return piece1,piece2

    for line in clean_data:
        i =  random.randint(1,10)
        sent1,sent2 = get_evaldata(line[1],line[0])
        if len(sent1) + 1 > hp.maxlen1: continue
        if len(sent2) + 1 > hp.maxlen2: continue

        if  i < 8:
            data_train1.append(line[1])
            data_train2.append(line[0])
        elif i < 9:
            data_eva1.append(line[1])
            data_eva2.append(line[2])

            fout1.write(line[1] + '\n')
            fout2.write(line[0] + '\n')


if __name__ == '__main__':
   logging.INFO("prepro begin...")
   hparams = Hparams()
   parser = hparams.parser
   hp = parser.parse_args()
   prepro(hp)
   logging.INFO("Done")