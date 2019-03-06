# -*- coding: utf-8 -*-
#最小角度回归（LARS）算法-添加10折交叉验证
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

#构建交叉验证循环决定最好的系数值
nxval = 10

#步数和步长
nSteps = 700
stepSize = 0.004

#初始化存储错误列表
errors = []
for i in range(nSteps):
    b = []
    errors.append([])

#10折交叉 测试集为十分之一的数据索引
for ixval in range(nxval):
    #决定测试和训练集合
    idxTest = [a for a in range(nrows) if a%nxval == ixval]
    idxTrain = [a for a in range(nrows) if a%nxval != ixval]

    # print(idxTest)
    # print(len(idxTest))
    # print(idxTrain)
    # print(len(idxTrain))

    #依据索引找到训练集与测试集的值
    xTrain = [xNormalized[r] for r in idxTrain]
    xTest = [xNormalized[r] for r in idxTest]
    labelTrain = [labelNormalized[r] for r in idxTrain]
    labelTest = [labelNormalized[r] for r in idxTest]

    #在训练数据集上构建LARS模型
    nrowsTrain = len(idxTrain)
    nrowsTest = len(idxTest)

    #初始化beta系数向量
    beta = [0.0] * ncols

    #初始化每一步的beta矩阵
    betaMat = []
    betaMat.append(list(beta))
    # print(betaMat)

    for i in range(nSteps):

        #计算残差
        residuals = [0.0] * nrows
        for j in range(nrowsTrain):
            # print("beta \n")
            # print(sum([xNormalized[j][k]*beta[k] for k in range(ncols)]))
            labelsHat = sum(xTrain[j][k] * beta[k] for k in range(ncols))
            # 残差
            residuals[j] = labelTrain[j] - labelsHat

        # print(residuals)

        #计算属性间相关性
        corr = [0.0] * ncols
        for j in range(ncols):
            corr[j] = sum([xTrain[k][j] * residuals[k] for k in range(nrowsTrain)]) / nrowsTrain
        iStar = 0
        #找corr里绝对值对大值
        corrStar = corr[0]
        for j in range(1,(ncols)):
            if abs(corrStar) < abs(corr[j]):
                iStar = j; corrStar = corr[j]
        beta[iStar] += stepSize * corrStar / abs(corrStar)
        betaMat.append(list(beta))
        if i == 508:
            print(betaMat[508])

        #利用系数计算预测值
        for j in range(nrowsTest):
            labelsHat = sum([xTest[j][k] * beta[k] for k in range(ncols)])
            err = labelTest[j] - labelsHat
            errors[i].append(err)

    # for i in range(ncols):
    #     coefCurve = [betaMat[k][i] for k in range(nSteps)]
    #     xaxis = range(nSteps)
    #     plot.plot(xaxis, coefCurve)
    # plot.xlabel('Step Taken')
    # plot.ylabel('coefficient values')
    # plot.show()
cvCurve = []
#MSE取平均值 errors是一个700*480的矩阵
for errVect in errors:
    mse = sum([x*x for x in errVect])/len(errVect)
    cvCurve.append(mse)

minMse = min(cvCurve)
minPt = [i for i in range(len(cvCurve)) if cvCurve[i] == minMse][0]
print("Minimum Mean Square Error",minMse)
print("Index of Minimum Mean Square Error", minPt)

xaxis = range(len(cvCurve))
plot.plot(xaxis, cvCurve)
plot.xlabel("Steps taken")
plot.ylabel("mean square error")
plot.show()


# for i in range(ncols):
#     coefCurve = [betaMat[k][i] for k in range(nSteps)]
#     xaxis =  range(nSteps)
#     plot.plot(xaxis, coefCurve)
# plot.xlabel('Step Taken')
# plot.ylabel('coefficient values')
# plot.show()
