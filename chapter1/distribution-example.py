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
def binom_pmf_test():
  '''
  为离散分布
  二项分布的例子：抛掷100次硬币，恰好两次正面朝上的概率是多少？
  '''
  n = 100#独立实验次数
  p = 0.5#每次正面朝上概率
  k = np.arange(0,100)#0-100次正面朝上概率
  binomial = stats.binom.pmf(k,n,p)
  print( binomial)#概率和为1
  print(sum(binomial))
  print( binomial[2])
  plt.plot(k, binomial,'o-')
  plt.title('Binomial: n=%i , p=%.2f' % (n,p),fontsize=15)
  plt.xlabel('Number of successes')
  plt.ylabel('Probability of success',fontsize=15)
  plt.show()

def  normal_distribution():
    '''
    正态分布是一种连续分布，其函数可以在实线上的任何地方取值。
    正态分布由两个参数描述：分布的平均值μ和方差σ2 。
    '''
    mu = 0  # mean
    sigma = 1  # standard deviation
    x = np.arange(-10, 10, 0.1)
    y = stats.norm.pdf(x, 0, 1)
    print(y)
    plt.plot(x, y)
    plt.title('Normal: $\mu$=%.1f, $\sigma^2$=%.1f' % (mu, sigma))
    plt.xlabel('x')
    plt.ylabel('Probability density', fontsize=15)
    plt.show()



if __name__ =="__main__":
    #binom_pmf_test() # 二项分布
    normal_distribution() # 正态分布