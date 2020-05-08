#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:Trie.py
# @Author: Michael.liu
# @Date:2020/5/6 16:41
# @Desc: this code is ....
import os
import codecs

class NULL(object):
    pass


class TrieNode:
    def __init__(self,value = NULL):
        self.value = value
        self.children = {}


class Trie:
    def __init__(self):
        self.root = TrieNode()


    def insert(self,key,value=None,sep=' '):
        elements = key if isinstance(key,list) else key.split(sep)
        node = self.root
        for e in elements:
            if not e: continue
            if  e not in node.children:
                child = TrieNode()
                node.children[e] = child
                node = child
            else:
                node = node.children[e]
        node.value = value

    def get(self,key,default= None,sep= ' '):
        elements = key if isinstance(key,list) else key.split(sep)
        node = self.root
        for e in elements:
            if  e not in node.children:
                return default
            node = node.children[e]
        return default if node.value is NULL else node.value

    def delete(self,key,sep =' '):
        elements = key if isinstance(key,list) else key.split(sep)
        return self.__delete(elements)

    def __delete(self,elements,node = None,i =0):
        node = node if node else self.root
        e = elements[i]
        if e in node.children:
            child_node = node.children[e]
            if len(elements) ==(i+1):
                if child_node.value is NULL: return False
                if len(child_node.children) == 0:
                    node.children.pop(e)
                else:
                    child_node.value = NULL
                return True
            elif self.__delete(elements,child_node,i+1):
                if len(child_node.children) ==0:
                    return node.children.pop(e)
                return True
        return False

    def shortest_prefix(self,key,default = NULL,sep = ' '):
        elements = key if isinstance(key,list) else key.split(sep)
        result = []
        node = self.root
        value = node.value
        for e in elements:
            if e in node.children:
                result.append(e)
                node = node.children[e]
                value = node.value
                if value is not NULL:
                    return sep.join(result)
            else:
                break
        if value is NULL:
            if default is not NULL:
                return default
            else:
                raise Exception("no item matches any prefix of the given key!")
        return sep.join(result)

    def longest_prefix(self,key,default = NULL, sep = ' '):
        elements = key if isinstance(key,list) else key.split(sep)
        results = []
        node = self.root
        value = node.value
        for e in elements:
            if e not in node.children:
                if value is not NULL:
                    return sep.join(results)
                elif default is not NULL:
                    return default
                else:
                    raise Exception("no item matches any prefix of the given key!")
            results.append(e)
            node = node.children[e]
            value = node.value
        if value is NULL:
            if default is not NULL:
                return default
            else:
                raise Exception("no item matches any prefix of the given key!")
        return sep.join(results)

    def longest_prefix_value(self,key,default=NULL,sep=' '):
        elements = key if isinstance(key,list) else key.split(sep)
        node = self.root
        value = node.value
        for e in elements:
            if e not in node.children:
                if value is not NULL:
                    return value
                elif default is not NULL:
                    return default
                else:
                    raise Exception("no item matches any prefix of the given key!")
            node = node.children[e]
            value = node.value
        if value is not NULL:
            return value
        if default is  not NULL:
            return default
        raise Exception("no item matches any prefix of the given key!")


    def longest_prefix_item(self, key, default=NULL, sep=' '):
        elements = key if isinstance(key, list) else key.split(sep)
        node = self.root
        value = node.value
        results = []
        for e in elements:
            if e not in node.children:
                if value is not NULL:
                    return (sep.join(results), value)
                elif default is not NULL:
                    return default
                else:
                    raise Exception("no item matches any prefix of the given key!")
            results.append(e)
            node = node.children[e]
            value = node.value
        if value is not NULL:
            return (sep.join(results), value)
        if default is not NULL:
            return (sep.join(results), default)
        raise Exception("no item matches any prefix of the given key!")


    def __collect_items(self, node, path, results, sep):
        if node.value is not NULL:
            results.append((sep.join(path), node.value))
        for k, v in node.children.iteritems():
            path.append(k)
            self.__collect_items(v, path, results, sep)
            path.pop()
        return results


    def items(self, prefix, sep=' '):
        elements = prefix if isinstance(prefix, list) else prefix.split(sep)
        node = self.root
        for e in elements:
            if e not in node.children:
                return []
            node = node.children[e]
        results = []
        path = [prefix]
        self.__collect_items(node, path, results, sep)
        return results


    def keys(self, prefix, sep=' '):
        items = self.items(prefix, sep)
        return [key for key, value in items]


def build_dic_trie(dic_path):
    trie = Trie()

    lines = codecs.open(dict_path,'r',encoding="utf-8")
    i = 0
    for line in lines:
        term = line.lower().strip().split("\t")[0]
        i = i+1
        trie.insert(term, i)
    return trie



if __name__ == "__main__":
    fileName = '../../data/chapter4/trie/CoreNatureDictionary.txt'
    print(os.path.abspath(fileName))
    dict_path = os.path.abspath(fileName)
    print("dic path:"+ dict_path)
    print("加载词典>>>>>>>>")
    trie = build_dic_trie(dict_path)

    print(trie.get(u"龚庄村"))
    print(trie.longest_prefix("龚庄村"))
    print(trie.longest_prefix("村"))
    print(trie.longest_prefix_value("村"))
    print(trie.longest_prefix_item("龚庄村"))


