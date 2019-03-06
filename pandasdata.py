__author__ = "susu"
import pandas as pd
from pandas import DataFrame
import numpy as np
import scipy.stats as stats
from math import sqrt
import sys
from pylab import  *
import matplotlib.pyplot as plot
# target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
# "databases/undocumented/connectionist-bench/sonar/sonar.all-data")
target_url = ("D:/Personal/Desktop/fatigue2.0.csv")
#从读取到pandas csv 数据 head=0 列标题为第一行数据；head = None 没有列标题；prefix 添加列标题前缀
fatigue = pd.read_csv(target_url, header=0, encoding="utf-8")
# print(fatigue.head())   #默认显示前5行数据
# print(fatigue.tail())   #默认显示最后5行数据

#数据集统计信息
summary = fatigue.describe()
min = summary.iloc[3,7]
max = summary.iloc[7,7]
nrows = len(fatigue.index)
#归一化
for i in range(nrows):
    dataRow = fatigue.iloc[i, 3:15]
    # labelColor = (fatigue.iloc[i, 18] - min)/(max - min)
    normTarget = (fatigue.iloc[i, 15] - min) / (max - min)
    labelColor = 1.0/(1.0+exp(-normTarget))
    dataRow.plot(color=plot.cm.RdYlBu(labelColor), alpha=0.5)

plot.xlabel("Attribute index")
plot.ylabel(("Attribute Values"))
plot.show()


#构建属性对的交会图
dataRow5 = fatigue.iloc[0:479, 3]
dataRow7 = fatigue.iloc[0:479, 4]
dataRow18 = fatigue.iloc[0:479, 15]
plot.scatter(dataRow5,dataRow18)
plot.xlabel("1 attribute")
plot.ylabel("2 attribute")
plot.show()

#构建箱线图
array = fatigue.iloc[0:479,0:15].values
boxplot(array)
plot.xlabel("属性索引")
plot.ylabel("分位数")
show()

#构建标准化箱线图,标准化与归一化不同
fatigueNormalize = fatigue.iloc[:,0:15]
for i in range(16):
    mean = summary.iloc[1, i]
    std = summary.iloc[2, i]
    fatigueNormalize.iloc[:,i:(i+1)] = (fatigueNormalize.iloc[:,i:(i+1)] -mean) /std
array2 = fatigueNormalize.values
boxplot(array2)
plot.xlabel("Attribute index")
plot.ylabel("Quatile Range - Normalized")
show()

#计算皮尔逊相关系数-不使用内部方法-可进一步熟悉皮尔逊相关系数的计算
mean5 = 0; mean7 = 0; mean18 = 0;
numElt = len(dataRow5)
for i in range(numElt):
    mean5 += dataRow5[i]/numElt    #均值
    mean7 += dataRow7[i]/numElt
    mean18 += dataRow18[i]/numElt
var5 = 0; var7 = 0; var18 = 0;
for i in range(numElt):
    var5 += (dataRow5[i] - mean5) * (dataRow5[i] - mean5)/numElt  #方差
    var7 += (dataRow7[i] - mean7) * (dataRow7[i] - mean7) / numElt
    var18 += (dataRow18[i] - mean18) * (dataRow18[i] - mean18) / numElt
corr57 = 0; corr518 = 0;
for i in range(numElt):
    corr57 += (dataRow5[i] - mean5) * (dataRow7[i] - mean7)/(sqrt(var5 * var7) * numElt)
    corr518 += (dataRow5[i] - mean5) * (dataRow18[i] - mean18)/(sqrt(var5 * var18) * numElt)

sys.stdout.write("Correlation between attribute 5 and 7 \n")
print(corr57)
sys.stdout.write("Correlation between attribute 5 and 18 \n")
print(corr518)

corFatigue = DataFrame(fatigue.corr())

#展示热图
plot.pcolor(corFatigue)
plot.show()


#显示属性的关联热图和关联矩阵
# X,Y,Z,H,A ,K,annual mean wind speed,V50,Ve50,turbulent intensity,air density,wind shear,
# inflow angle,power,blade diameter,weight tower,number tower,blade type,BR_Mx_m4
# print(fatigue.iloc[:,['H','A','K','annual mean wind speed','V50','Ve50',
#                                     'turbulent intensity','air density','wind shear',
#                                     'inflow angle','blade diameter','weight tower','BR_Mx_m4']])
corFati = DataFrame(fatigue.loc[:,['H','A','K','annual mean wind speed','V50','Ve50',
                                    'turbulent intensity','air density','wind shear',
                                    'inflow angle','blade diameter','weight tower','BR_Mx_m4']].corr())
print(corFati)
plot.pcolor(corFati)
plot.show()
# doc = open('out.txt','w')
#
# print(np.round(rocksVmines.corr(method="pearson"),2))
# print(summary,file=doc)
# doc.close()