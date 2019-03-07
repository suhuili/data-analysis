# -*- coding: utf-8 -*-
import pandas as pd
from scipy.interpolate import lagrange #导入拉格朗日插值函数

target_url = ("D:/Personal/Desktop/zhengqi_train.txt")
zhengqi_train = pd.read_table(target_url, header=0, encoding="utf-8")

#数据探索-缺失值检测
explore = zhengqi_train.describe(percentiles = [], include = 'all').T #包括对数据的基本描述，percentiles参数是指定计算多少的分位数表（如1/4分位数、中位数等）；T是转置，转置后更方便查阅
explore['null'] = len(zhengqi_train)-explore['count'] #describe()函数自动计算非空值数，需要手动计算空值数
print(explore)

#数据探索-异常值检测
#箱线图
import matplotlib.pyplot as plt #导入图像库
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

plt.figure() #建立图像
p = zhengqi_train.boxplot(return_type='dict') #画箱线图，直接使用DataFrame的方法
x = p['fliers'][9].get_xdata() # 'flies'即为异常值的标签
y = p['fliers'][9].get_ydata()
y.sort() #从小到大排序，该方法直接改变原对象

#用annotate添加注释
#其中有些相近的点，注解会出现重叠，难以看清，需要一些技巧来控制。
#以下参数都是经过调试的，需要具体问题具体调试。
for i in range(len(x)):
  if i>0:
    plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.05 -0.8/(y[i]-y[i-1]),y[i]))
  else:
    plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.08,y[i]))

plt.show() #展示箱线图

#异常值处理，拉格朗日插值不准确，利用上下均值代替异常点
zhengqi_train['V9'][zhengqi_train['V9'] < -9] = None #过滤异常数据

# 拉格朗日插值
#自定义列向量插值函数
#s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def ployinterp_column(s, n, k=5):
  y = s[list(range(n-k, n)) + list(range(n+1, n+1+k))] #取数
  y = y[y.notnull()] #剔除空值
  return lagrange(y.index, list(y))(n) #插值并返回插值结果

#逐个元素判断是否需要插值
for i in zhengqi_train.columns:
  for j in range(len(zhengqi_train)):
    if (zhengqi_train[i].isnull())[j]: #如果为空即插值。
      # zhengqi_train[i][j] = ployinterp_column(zhengqi_train[i], j)
        zhengqi_train[i][j] = (zhengqi_train[i][j-1]+zhengqi_train[i][j+1])/2
        print(zhengqi_train[i][j])