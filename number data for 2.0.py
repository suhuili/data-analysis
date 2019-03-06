# -*- coding: utf-8 -*-
import pandas as pd
from pandas import DataFrame
from pylab import *
import matplotlib.pyplot as plot

target_url = ("D:/Personal/Desktop/fatigue2.0.csv")
fatigue = pd.read_csv(target_url, header=0, encoding="utf-8")
summary = fatigue.describe()
print(summary)
fatigueNormarlized = fatigue
ncols = len(fatigueNormarlized.columns)

#标准化
for i in range(ncols):
    mean = summary.iloc[1,i]
    sd = summary.iloc[2, i]
    fatigueNormarlized.iloc[:,i:(i+1)] = (fatigueNormarlized.iloc[:,i:(i+1)] - mean)/ sd
array = fatigueNormarlized.values
boxplot(array)
plot.xlabel("Attribute index")
plot.ylabel("Quartile Ranges - Normalized")
show()

#构建平行坐标图
nrows = len(fatigue.index)
BladeCol = len(summary.columns)
meanBlade = summary.iloc[1, BladeCol - 1]
sdBlabe = summary.iloc[2, BladeCol -1]
nDataCol = len(fatigue.columns) -1
for i in range(nrows):
    dataRow = fatigue.iloc[i, 3:nDataCol]
    normTarget = (fatigue.iloc[i, nDataCol] - meanBlade) / sdBlabe
    labelColor = 1.0/(1.0 + exp(-normTarget))
    dataRow.plot(color = plot.cm.RdYlBu(labelColor), alpha = 0.5)

plot.xlabel("Attribute Index")
plot.ylabel("Attribute Values")
plot.show()

for i in range(nrows):
    dataRow = fatigueNormarlized.iloc[i, 3:nDataCol]
    normTarget = fatigueNormarlized.iloc[i, nDataCol]
    labelColor = 1.0/(1.0 + exp(-normTarget))
    dataRow.plot(color=plot.cm.RdYlBu(labelColor),alpha = 0.5)
plot.xlabel("Attribute Index")
plot.ylabel("Attribute Values")
plot.show()

corFatigue = DataFrame(fatigue.corr())

#展示热图
plot.pcolor(corFatigue)
plot.show()