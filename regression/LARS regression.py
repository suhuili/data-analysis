# -*- coding: utf-8 -*-
#最小角度回归（LARS）算法
__author__ = "susu"
import numpy
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plot
import csv
import sys


target_url = ("D:/Personal/Desktop/fatigue2.0_2y.csv")
data = csv.reader(open(target_url, encoding='utf-8'))
xList = []
labels = []
names = []
firstLine = True

for index,line in enumerate(data):
    #逗号分割
    # row = line.decode().strip().split(',')
    # print(row)
    # xList.append(row)
    if index == 0:
        # labels.append(line)
        names = line
        continue
    floatline = [float(num) for num in line]
    labels.append(float(line[-1]))   #得到预测标签的列
    floatline.pop()
    floatline.pop()
    floatline.pop()
    xList.append(floatline)

nrows = len(xList) # 行数
ncols = len(xList[0]) # 列数

#计算每一列的平均值、方差、标准差
xMeans =[]
xSD = []
for i in range(ncols):
    col = [xList[j][i] for j in range(nrows)]
    mean = sum(col)/nrows
    xMeans.append(mean)
    colDiff = [(xList[j][i] - mean) for j in range(nrows)]
    sumSq = sum([colDiff[i] * colDiff[i] for i in range(nrows)])
    stdDev = sqrt(sumSq/nrows)
    xSD.append(stdDev)
# print(xMeans)
# print(xSD)

#标准化 均值为0 方差为1
#这里的数据都为输入属性值
xNormalized = []
for i in range(nrows):
    #行数据标准化
    rowNormalized = [(xList[i][j] - xMeans[j])/xSD[j] for j in range(ncols)]
    xNormalized.append(rowNormalized)

#标准化标签值
meanLabel = sum(labels)/nrows
sdLabel = sqrt(sum([(labels[i] - meanLabel) * (labels[i] - meanLabel) for i in range(nrows)])/nrows)
labelNormalized = [(labels[i] - meanLabel) / sdLabel for i in range(nrows)]

#初始化beta系数向量
beta = [0.0] * ncols

#初始化每一步的beta矩阵
betaMat = []
betaMat.append(list(beta))
# print(betaMat)

nSteps = 700
stepSize = 0.004
for i in range(nSteps):
    #计算残差
    residuals = [0.0] * nrows
    for j in range(nrows):
        # print("beta \n")
        # print(sum([xNormalized[j][k]*beta[k] for k in range(ncols)]))
        labelsHat = sum(xNormalized[j][k] * beta[k] for k in range(ncols))
        # 残差
        residuals[j] = labelNormalized[j] - labelsHat

    # print(residuals)

    #计算属性间相关性
    corr = [0.0] * ncols
    for j in range(ncols):
        corr[j] = sum([xNormalized[k][j] * residuals[k] for k in range(nrows)]) / nrows
    iStar = 0
    #找corr里绝对值对大值
    corrStar = corr[0]
    for j in range(1,(ncols)):
        if abs(corrStar) < abs(corr[j]):
            iStar = j; corrStar = corr[j]
    beta[iStar] += stepSize * corrStar / abs(corrStar)
    betaMat.append(list(beta))
for i in range(ncols):
    coefCurve = [betaMat[k][i] for k in range(nSteps)]
    xaxis =  range(nSteps)
    plot.plot(xaxis, coefCurve)
plot.xlabel('Step Taken')
plot.ylabel('coefficient values')
plot.show()
