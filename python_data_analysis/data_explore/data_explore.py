# -*- coding: utf-8 -*-
import pandas as pd


target_url = ("D:/Personal/Desktop/zhengqi_train.txt")
zhengqi_train = pd.read_table(target_url, header=0, encoding="utf-8")

#数据探索-缺失值检测
explore = zhengqi_train.describe(percentiles = [], include = 'all').T #包括对数据的基本描述，percentiles参数是指定计算多少的分位数表（如1/4分位数、中位数等）；T是转置，转置后更方便查阅
explore['null'] = len(zhengqi_train)-explore['count'] #describe()函数自动计算非空值数，需要手动计算空值数
print(explore)