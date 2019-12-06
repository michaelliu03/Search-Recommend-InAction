#!/usr/bin/env python
#-*-coding:utf-8-*-
# @File:distributiion-example.py
# @Author: Michael.liu
# @Date:2019/2/12
import numpy as np

from scipy import stats
import matplotlib.pyplot as plt
#####################
#二项分布
#####################
def test_binom_pmf():
  '''
  为离散分布
  二项分布的例子：抛掷10次硬币，恰好两次正面朝上的概率是多少？
  '''
  n = 10#独立实验次数
  p = 0.5#每次正面朝上概率
  k = np.arange(0,11)#0-10次正面朝上概率
  binomial = stats.binom.pmf(k,n,p)
  print( binomial)#概率和为1
  print(sum(binomial))
  print( binomial[2])
  plt.plot(k, binomial,'o-')
  plt.title('Binomial: n=%i , p=%.2f' % (n,p),fontsize=15)
  plt.xlabel('Number of successes')
  plt.ylabel('Probability of success',fontsize=15)
  plt.show()

test_binom_pmf()