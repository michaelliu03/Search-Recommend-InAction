#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 
# @desc :

import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_size',default=32000,type=int)
    parser.add_argument('--vocab',default='amazon/bpe.vocab',help="vocabulary file")

    #train
    parser.add_argument('--amazon_origin',default='amazon/amazon_product.txt',help='amazon origin data')
    parser.add_argument('--train1',default='amazon/train1.txt',help='training segmented data')   # keywords
    parser.add_argument('--train2',default='amazon/train2.txt',help='training segmented data')   # title
    parser.add_argument('eval1',default='amazon/eval1.txt',help ='evaluation segmented data')
    parser.add_argument('--eval2', default='amazon/eval2.txt', help="evaluation segmented data")


    #test
    parser.add_argument('--test1',default='amazon/test1_.txt',help="test segment")
    parser.add_argument('--test2',default='amazon/test2_.txt',help="test segment")
    parser.add_argument('--ckpt',default='model/',help="checkpoint file path")