#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:build_index.py
# @Author: Michael.liu
# @Date:2019/2/12
# @Desc: 主要是为了建立搜索引擎中的index,分词器可以用jieba，也可以用hanlp，
#        因为本人对hanlp比较熟悉，所以采用了hanlp，还有这里选用了sqlite3引擎
#        其实本人更喜欢把这部分数据放到mysql或者hbase上,整个代码在改写过程中遇到了一些坑，
#        欢迎读者朋友可以自行尝试！！！^_^
#        在爬虫部分中的代码已经看到了抓取网页的代码，这里就不再赘述，只是看一下采集到的新闻格式如下：
'''
<?xml version='1.0' encoding='utf-8'?>
<doc><id>1</id><url>http://www.chinanews.com/gn/2020/04-08/9151413.shtml</url><title>国务院联防联控机制：发现无症状感染者应于2小时内网络直报</title><datetime>2020-4-8 23:57:00</datetime><body>	(抗击新冠肺炎)国务院联防联控机制：发现无症状感染者应于2小时内网络直报
	中新社北京4月8日电据中国政府网8日晚发布的消息，国务院联防联控机制近日印发《新冠病毒无症状感染者管理规范》，规定各级各类医疗卫生机构发现无症状感染者，应当于2小时内进行网络直报。
	根据《规范》，无症状感染者具有传染性，存在着传播风险。
	《规范》提出加强对无症状感染者的监测和发现：一是对新冠肺炎病例的密切接触者医学观察期间的主动检测；二是在聚集性疫情调查中的主动检测；三是在新冠肺炎病例的传染源追踪过程中对暴露人群的主动检测；四是对部分有境内外新冠肺炎病例持续传播地区旅居史人员的主动检测；五是在流行病学调查和机会性筛查中发现的相关人员。
	《规范》要求规范无症状感染者的报告。各级各类医疗卫生机构发现无症状感染者，应当于2小时内进行网络直报。县级疾控机构接到发现无症状感染者报告后，24小时内完成个案调查，并及时进行密切接触者登记，将个案调查表或调查报告及时通过传染病报告信息管理系统进行上报。
	《规范》要求加强对无症状感染者的管理。集中医学观察满14天且连续两次标本核酸检测呈阴性者(采样时间至少间隔24小时)可解除集中医学观察，核酸检测仍为阳性且无临床症状者需继续集中医学观察。
	《规范》要求有针对性加大筛查力度，将检测范围扩大至已发现病例和无症状感染者的密切接触者。做好对重点地区、重点人群、重点场所的强化监测，一旦发现无症状感染者应当集中隔离医学观察。
	《规范》指出，无症状感染者具有传播隐匿性、症状主观性、发现局限性等特点，国家支持开展无症状感染者传染性、传播力、流行病学等科学研究。加强与世界卫生组织等有关国家和国际组织的信息沟通、交流合作，适时调整诊疗方案和防控方案。(完)
【编辑:张楷欣】
</body></doc>
'''
# 所以doc包括docid，date_time，content
#       doc1   doc2    doc3
#
#

import os

from os import listdir
from pyhanlp import *
import sqlite3
import xml.etree.ElementTree as ET
import configparser
from pyhanlp import *





'''
布尔检索模型,tf - 词项频率； ld - 文本长度
'''
class Doc:
    docid = 0
    date_time = ''
    tf =0
    ld = 0
    def __init__(self,docid,date_time,tf,ld):
        self.docid = docid
        self.date_time = date_time
        self.tf = tf
        self.ld = ld
    def  __repr__(self):
        return(str(self.docid) + '\t' + self.date_time + '\t' + str(self.tf) + '\t'+str(self.ld))
    def __str__(self):
        return (str(self.docid) + '\t' + self.date_time + '\t' + str(self.tf) + '\t' + str(self.ld))


class SearchIndex:
     stop_words = set()
     postings_lists= {}

     config_path = ''
     config_encoding = ''

     def __init__(self,config_path,config_encoding):
        self.config_path =config_path
        self.config_encoding = config_encoding
        config = configparser.ConfigParser()
        config.read(config_path, config_encoding)
        print(config_path + "\t" + config_encoding)
        file_path = os.path.join(os.path.dirname(__file__),config['DEFAULT']['stop_words_path'])
        file_encoding =config['DEFAULT']['stop_words_encoding']
        f = open(file_path, encoding=file_encoding)
        words = f.read()
        self.stop_words = set(words.split('\n'))



     def is_number(self, s):
         try:
             float(s)
             return True
         except ValueError:
             return False

     def clean_list(self, seg_list):
         cleaned_dict = {}
         n = 0
         for i in seg_list:
             i = i.strip().lower()

             if i != '' and not self.is_number(i) and i not in self.stop_words:
                 n = n + 1
                 if i in cleaned_dict:
                     cleaned_dict[i] = cleaned_dict[i] + 1
                 else:
                     cleaned_dict[i] = 1
 #        print(cleaned_dict)
         return n, cleaned_dict

     '''
        这里写了sqlite ，第一次操作sqlite
     '''

     def write_postings_to_db(self, db_path):
         conn = sqlite3.connect(db_path)
         c = conn.cursor()

         c.execute('''DROP TABLE IF EXISTS postings''')
         c.execute('''CREATE TABLE postings
                       (term TEXT PRIMARY KEY, df INTEGER, docs TEXT)''')

         for key, value in self.postings_lists.items():
             doc_list = '\n'.join(map(str, value[1]))
             t = (key, value[0], doc_list)
             c.execute("INSERT INTO postings VALUES (?, ?, ?)", t)

         conn.commit()
         conn.close()

     def new_seg(self,content):
         ret = []
         seg_list = HanLP.segment(content)
         for term in seg_list:
             word = str(term.word)
             # 判断新
             if word =='' or word =='\r' or word =='\t\n' or word =='\n' or word =='\t':
                 continue
             else:
                 ret.append(word)
         return ret


     def build_postings_index(self):
         config = configparser.ConfigParser()
         config.read(self.config_path, self.config_encoding)
         files = listdir(config['DEFAULT']['doc_dir_path'])
         AVG_L = 0
         HanLP.Config.ShowTermNature = False # 关闭词性
         for i in files:
             root = ET.parse(config['DEFAULT']['doc_dir_path'] + i).getroot()
             title = root.find('title').text
             body = root.find('body').text
             docid = int(root.find('id').text)
             date_time = root.find('datetime').text
             content  = "".join(title + '。' + body)
             seg_py_list = self.new_seg(content)
             ld, cleaned_dict = self.clean_list(seg_py_list)
             AVG_L = AVG_L + ld

             for key, value in cleaned_dict.items():
                 d = Doc(docid, date_time, value, ld)
                 if key in self.postings_lists:
                     self.postings_lists[key][0] = self.postings_lists[key][0] + 1  # df++
                     self.postings_lists[key][1].append(d)
                 else:
                     self.postings_lists[key] = [1, [d]]  # [df, [Doc]]
         AVG_L = AVG_L / len(files)
         config.set('DEFAULT', 'N', str(len(files)))
         config.set('DEFAULT', 'avg_l', str(AVG_L))
         with open(self.config_path, 'w', encoding=self.config_encoding) as configfile:
             config.write(configfile)
         db_path = os.path.join(os.path.dirname(__file__), config['DEFAULT']['db_path'])

         self.write_postings_to_db(db_path)


if __name__ == "__main__":
    filename = os.path.join(os.path.dirname(__file__), 'config.ini')
    ir = SearchIndex(filename, 'utf-8')
    ir.build_postings_index()
    print("finished!")

