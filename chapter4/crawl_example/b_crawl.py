#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2020/12/17 10:07 
# @Author : michael.liu 
# @Version：V 0.1
# @File : b_crawl.py
# @desc :

from bs4 import BeautifulSoup
import urllib.request
import xml.etree.ElementTree as ET
import configparser
import datetime
from datetime import timedelta, date
import time
import urllib.parse
import socket
from socket import timeout


user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
headers = {'User-Agent': user_agent}



def get_one_page_news():
    #    page_url='http://www.chinanews.com/scroll-news/2019/0801/news.shtml'
    root = 'https://m.btime.com/item/437cl1f7gbt9jlbr0ja5rlq36d4'
    req = urllib.request.Request(root, headers=headers)

    try:
        response = urllib.request.urlopen(req, timeout=10)
        html = response.read()
    except socket.timeout as err:
        print('socket.timeout')
        print(err)
        return []
    except Exception as e:
        print("-----%s:%s %s-----" % (type(e), e.reason, page_url))
        return []

    soup = BeautifulSoup(html, "html.parser")  # http://www.crummy.com/software/BeautifulSoup/bs4/doc.zh/

    news_pool = []
    news_list = soup.find('div', class_="content_list")
    items = news_list.find_all('li')
    for i, item in enumerate(items):
        #        print('%d/%d'%(i,len(items)))
        if len(item) == 0:
            continue

        a = item.find('div', class_="dd_bt").find('a')
        title = a.string
        url = a.get('href')
        if root in url:
            url = url[len(root):]

        category = ''
        try:
            category = item.find('div', class_="dd_lm").find('a').string
        except Exception as e:
            continue

        if category == '图片':
            continue

        year = url.split('/')[-3]
        date_time = item.find('div', class_="dd_time").string
        date_time = '%s-%s:00' % (year, date_time)

        news_info = [date_time, "http://www.chinanews.com" + url, title]
        news_pool.append(news_info)
    return news_pool


# def get_news_pool(start_date, end_date):
#     news_pool = []
#     delta = timedelta(days=1)
#     while start_date <= end_date:
#         date_str = start_date.strftime("%Y/%m%d")
#         page_url = 'http://www.chinanews.com/scroll-news/%s/news.shtml' % (date_str)
#         print('Extracting news-2020-0503-part1-2020-04-26-part2-2020-04-26-part1-2020-04-26-2020-04-22-part2-2020-04-22-part1-2020-04-21.4.20 urls at %s' % date_str)
#         news_pool += get_one_page_news(page_url)
#         #        print('done')
#         start_date += delta
#     return news_pool

def process():
    get_one_page_news()


if __name__ =='__main__':
    #start_time = datatime.datetime.now()
    process()
    #end_time = datetime.datetime.now()
    #print("time loss", end_time - start_time)
    #print("Hello World!")