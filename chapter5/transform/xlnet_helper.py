#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 
# @desc :
import requests,json
import logging

import xlnet
import sentencepiece as spm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')




class XlnetHelper(object):
    def __init__(self, url= 'http://127.0.0.1'):
        self.url = url


    def embedding_data(self, data):
        body = {"texts": data}
        r =requests.post(self.url, data=json.dumps(body))
        code = r.status_code
        if code == 200:
            return r.json()
        else:
            logging.error(r.reason)
            return []


if __name__ == '__main__':
    xlnet = XlnetHelper()
    data = ['Replace me by any text you Beach',
            'Replace me by any text you you you']
    data = ['Summer Short Beach Wedding Dresses 2019 Hot Sales Simple Style Cap Sleeve Knee Length A-Line Tulle Lace Bridal Gowns Vestidos de Novia W262',
            'Vintage Lace Appliques Cap Sleeves Wedding Dresses Illusion Back Sheer Neck Bridal Gowns Covered Buttons Wedding Gowns Vestido de Novias',
            '2020 Long Sleeve Lace Ball Gown Wedding Dresses Robe De Mariage Applique Vestido De Noiva De Renda Luxury Bridal Gowns HY230',
            'Vintage Short Cap Sleeves Wedding Dresses 2021 Lace Applique Ribbon Bow Covered Buttons Back Tulle Sweep Train Custom Made Chapel Bride Gown vestido de novia',
            'Vintage Champagne Lace Bohemian Wedding Dress A Line Cap Sleeve Backless Bow Sash Wedding Bridal Gown Vestidos de Novia 2020',
            'Bohemian Beach Lace Wedding Dress Boho Sexy 2022 Long Sleeve Backless V Neck Court Train Bridal Gowns',
            '2019 Bohemian Sexy Beach Wedding Dresses Bridal Gowns Sheer Deep V Neck 3D Floral Appliqued Lace Backless Country Plus Size Wedding Dress',
            'Vintage A-line New Designer Wedding Dress Arabic Buttons Short Sleeve Lace Off Shoulders Garden Long Bridal Gown Custom Made Plus Size']
    res = xlnet.embedding_data(data)
    print(res)
    # d1 = res[1]
    # s = 0
    # for d in d1:
    #     s+=d*d
    # print(s)