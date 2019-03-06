# -*- coding: utf-8 -*-
# 岭回归，依据alpha大小来限制最小二乘法过拟合
__author__ = "susu"
import numpy
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt
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
    labels.append(float(line[-3]))   #得到预测标签的列
    floatline.pop()
    floatline.pop()
    floatline.pop()
    xList.append(floatline)


indices = range(len(xList))
xListTest = [xList[i] for i in indices if i%3 == 0]
xListTrain = [xList[i] for i in indices if i%3 != 0]
labelsTest = [labels[i] for i in indices if i%3 == 0]
labelsTrain = [labels[i] for i in indices if i%3 != 0]

xTrain = numpy.array(xListTrain)
yTrain = numpy.array(labelsTrain)
xTest = numpy.array(xListTest)
yTest = numpy.array(labelsTest)

alphaList = [0.1** i for i in [0, 1, 2, 3, 4, 5, 6]]

rmsError = []
for alph in alphaList:
    fatigueModel = linear_model.Ridge(alpha=alph)
    fatigueModel.fit(xTrain, yTrain)
    # norm(x,ord = 2) l2范数，平方和再开方
    rmsError.append(numpy.linalg.norm((yTest-fatigueModel.predict(xTest)),2)/sqrt(len(yTest)))

print("RMS ERROR        alpha")
for i in range(len(rmsError)):
    print(rmsError[i], alphaList[i])

#输出样本外alpha 错误曲线
x = range(len(rmsError))
plt.plot(x, rmsError, 'k')
plt.xlabel('-log(alpha)')
plt.ylabel('Error (RMS)')
plt.show()

#依据最好的alpha值构建样本外错误直方图
#识别最小值的索引，重新训练此alpha值下的样本
indexBest = rmsError.index(min(rmsError))
alph = alphaList[indexBest]
print("\n" + "最优alpha")
print(alph)
fatigueModel = linear_model.Ridge(alpha=alph)
print("\n" + "系数")
fatigueModel.fit(xTrain, yTrain)
print(fatigueModel.coef_)
errorVector = yTest-fatigueModel.predict(xTest)
print("\n" + "错误向量")
print(errorVector)
plt.hist(errorVector)
plt.xlabel("Bin Boundaries")
plt.ylabel("Counts")
plt.show()

plt.scatter(fatigueModel.predict(xTest), yTest, s=100, alpha=0.10)
plt.xlabel('Predicted Fatigue ')
plt.ylabel('Actual fatigue')
plt.show()

