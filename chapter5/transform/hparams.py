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
    parser.add_argument('--vocab',default='./amazon/bpe.vocab',help="vocabulary file")

    #train
    parser.add_argument('--amazon_origin',default='./amazon/amazon_product.txt',help='amazon origin data')
    parser.add_argument('--train1',default='./amazon/train1.txt',help='training segmented data')   # keywords
    parser.add_argument('--train2',default='./amazon/train2.txt',help='training segmented data')   # title
    parser.add_argument('--eval1',default='./amazon/eval1.txt',help ='evaluation segmented data')
    parser.add_argument('--eval2', default='./amazon/eval2.txt', help="evaluation segmented data")

    # test
    parser.add_argument('--test1', default='./amazon/test1_.txt', help="test segment")
    parser.add_argument('--test2', default='./amazon/test2_.txt', help="test segment")
    parser.add_argument('--ckpt', default='./amazon/model/', help="checkpoint file path")
    parser.add_argument('--ckpt_pb', default='./amazon/model_pb/', help="checkpoint file path")

    parser.add_argument('--testdir', default="./amazon/test/", help="test result dir")
    parser.add_argument('--evaldir', default="./amazon/eval/", help="evaluation dir")

    # training scheme
    parser.add_argument('--batch_size', default=64, type=int)  ### 128
    parser.add_argument('--eval_batch_size', default=64, type=int)  ### 128
    parser.add_argument('--test_batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")  ### 0.0003
    parser.add_argument('--warmup_steps', default=1000, type=int)  ### 4000
    parser.add_argument('--num_epochs', default=15, type=int)  ### 20

    # model params
    parser.add_argument('--d_model', default=512, type=int, help="hidden dimension of encoder/decoder")  ### 512
    parser.add_argument('--d_ff', default=1024, type=int, help="hidden dimension of feedforward layer")  ### 2048
    parser.add_argument('--num_blocks', default=3, type=int, help="number of encoder/decoder blocks")  ### 6
    parser.add_argument('--num_heads', default=4, type=int, help="number of attention heads")  ### 8
    parser.add_argument('--maxlen1', default=50, type=int, help="maximum length of a source sequence")  ### 100
    parser.add_argument('--maxlen2', default=50, type=int, help="maximum length of a target sequence")  ### 100
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float, help="label smoothing rate")