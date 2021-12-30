#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/4/8 19:07
# @Author :'liuyu'
# @Version：V 0.1
# @File : 
# @desc :
#from __future__ import unicode_literals

import logging
import datetime
import os
import sys
import time


import elasticsearch
import elasticsearch.helpers
from elasticsearch.client.utils import _make_path,query_params,SKIP_IN_PATH

if os.getenv(u"UTILITY_ENV",None) in (u'dev','qa'):
    online_es_host = [{u"host": u"10.110.13.101", u"port": 9201}, {u"host": u"10.110.14.6", u"port": 9201}]
    offline_es_host = [{u"host": u"10.110.13.101", u"port": 9201}, {u"host": u"10.110.14.6", u"port": 9201}]
    headers = {"Authorization": "basic dHVpamlhbjA6RkVJQHR1aWppYW4="}
else:
    online_es_host = [
        {u"host": u"192.168.30.141", u"port": 2811}, {u"host": u"192.168.30.142", u"port": 2811},
        {u"host": u"192.168.30.143", u"port": 2811}
    ]
    offline_es_host = [
        {u"host": u"192.168.30.169", u"port": 2813}, {u"host": u"192.168.30.170", u"port": 2813},
        {u"host": u"192.168.30.171", u"port": 2813}, {u"host": u"192.168.30.177", u"port": 2813}
    ]
    headers = {"Authorization": "basic dHVpamlhbjA6RkVJQHR1aWppYW4="}

logger = logging.getLogger(__name__)

def action_callback(line):
    return ({line[u"action"]: {u"_index": line[u"_index"], u"_type": line[u"_type"], u"_id": line[u"_id"]}},
            line[u"source"])

class Search(object):
    def __init__(self,is_online=True, hosts=None, use_ssl=False, verify_certs=False, user_headers=None):
        """
        :param is_online
        :param hosts
        :param use_ssl
        :param verify_certs
        :param headers:
        """

        if is_online:
            if hosts is None:
                self.es_host = online_es_host
            else:
                self.es_host = hosts
        else:
            if hosts is None:
                self.es_host = offline_es_host
            else:
                self.es_host = hosts

        self.use_ssl = use_ssl
        self.verify_certs = verify_certs
        if user_headers is None:
            self.user_headers = headers
        else:
            self.user_headers = user_headers

        self.es_client = elasticsearch.Elasticsearch(hosts=self.es_host, use_ssl=self.use_ssl,
                                                     verify_certs=self.verify_certs,
                                                     headers=self.user_headers)
        
    def get_json(self, index, doc_type, _id, field_name, data, action):
        data_json = {u"_index": index, u"_type": doc_type, u"_id": _id, u"action": action}
        if field_name == u"doc":
            data_json[u"source"] = {u"doc": data, u"doc_as_upsert": True}
            # data_json[u"source"] = data
        else:
            data_json[u"source"] = {u"doc": {field_name: data}, u"doc_as_upsert": True}
            # data_json[u"source"] = {field_name: data}
        now = datetime.datetime.now().strftime(u"%Y-%m-%d %H:%M:%S")
        data_json[u"source"][u"doc"][u"update_time"] = now

        return data_json

    def bulk(self, data, chunk_size=400, timeout=60, retry=3):
        while retry > 0:
            retry -= 1
            try:
                errors = elasticsearch.helpers.bulk(client=self.es_client, actions=data, chunk_size=chunk_size,
                                                    expand_action_callback=action_callback, timeout="-1s",
                                                    request_timeout=timeout)
                return errors
            except elasticsearch.ConnectionTimeout as e:
                print(e, u"time out. sleep 120s for retry............", sys.exc_info()[2].tb_frame.f_back)
                time.sleep(120)
            except Exception as e:
                print(u"index data error: %s" % str(e))
                print(sys.exc_info()[2].tb_frame.f_back)
                import traceback
                traceback.print_exc()
                return []
        return []

    @query_params(u'_source', u'_source_exclude', u'_source_include',
                  u'allow_no_indices', u'analyze_wildcard', u'analyzer', u'conflicts',
                  u'default_operator', u'df', u'expand_wildcards', u'from_',
                  u'ignore_unavailable', u'lenient', u'preference', u'q', u'refresh',
                  u'request_cache', u'requests_per_second', u'routing', u'scroll',
                  u'scroll_size', u'search_timeout', u'search_type', u'size', u'slices',
                  u'sort', u'stats', u'terminate_after', u'timeout', u'version',
                  u'wait_for_active_shards', u'wait_for_completion')

    def delete_by_query(self, index, body, doc_type=None, **kwargs):
        """
        Delete all documents matching a query.
        https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-delete-by-query.html
        """
        for param in (index, body):
            if param in SKIP_IN_PATH:
                raise ValueError(u"Empty value passed for a required argument.")
        return self.es_client.delete_by_query(index=index, doc_type=doc_type, body=body, **kwargs)


if __name__ == '__main__':
    es_client = Search()
    data = [{u'action': u'update',
             u'source': {
                 u'doc': {u'skill_words': u'java spring 程序'},
                 u'doc_as_upsert': True},
             u'_type': u'_doc', u'_id': u'25231',
             u'_index': u'cobra_job_online'},

            {u'action': u'update',
             u'source': {
                 u'doc': {u'industry': u'010', u'job_status': u'1'},
                 u'doc_as_upsert': True},
             u'_type': u'_doc', u'_id': u'25231',
             u'_index': u'cobra_job_online'}]

    import json

    print(json.dumps(data, ensure_ascii=False))
    errors = es_client.bulk(data, chunk_size=2)
    print(errors)
    result = es_client.delete_by_query("cobra_job_online", doc_type="_doc", body={
        "query": {
            "match": {
                "skill_words": "java"
            }
        }
    })
    print(result)