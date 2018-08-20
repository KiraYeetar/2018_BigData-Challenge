-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import metrics
'''
一些基础的工具或环境函数
'''

def data_path():
    return '/mnt/datasets/fusai/'

def basic_path():
    return '/home/kesci/work/basic/'

def features_path():
    return '/home/kesci/work/features/'

def split_data(data, columns, start_day, end_day):
    data = data[(data[columns] >= start_day) & (data[columns] <= end_day)]
    return data 

'''
下面返回的lsit说明下：
以1-10天为起始特征区间，用于返回需要划多少天
例如划1-18，就返回18-10=8，
划测试集1-30，就返回30-10=20
需要少划几个对应修改就好
... 要修改起始特征区间，修改下面的 ups 和 downs 函数
''' 
def features_addday_list():
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]

def ups():
    return 1

def downs():
    return 10


