#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 这个类是一个封装不错的HiveClient
# @desc :

import os
import subprocess
import tempfile
import shutil

class HiveClient(object):
    def fetch(self,sql,output_path,sep='\t',title=None):
        cur_dir = os.path.dirname(os.path.abspath(output_path))
        path = tempfile.mkdtemp(dir = cur_dir)
        try:
            hql = """ insert overwrite local directory '%s' row format delimited fields terminated by '%s' %s 
            """ % (path, sep, sql)
            subprocess.check_call(['hive','-e',"%s" % hql])
            ommited_count = 0
            with open(output_path,'w',encoding='utf8') as target_fp:
                if title is not None:
                    target_fp.write(title.strip())
                    target_fp.write("\n")
                for txt_file in os.listdir(path):
                    if txt_file.endswith(".crc"):
                        continue
                    with open(os.path.join(path,txt_file),'r',encoding='utf-8') as fp:
                        try:
                            for line in fp:
                                target_fp.write(line)
                        except UnicodeDecodeError:
                            ommited_count += 1
        finally:
            shutil.rmtree(path)

