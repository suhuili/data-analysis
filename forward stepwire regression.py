# -*- coding: utf-8 -*-
__author__ = "susu"
#前向回归
import numpy
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt
import csv
import sys

def xattrSelect(x, idxSet):
    # 将输入变量作为矩阵，并返回输入变量的子集组合,返回类型为List
    # 在idxSet中的列
    xOut = []
    for row in x:
        xOut.append([row[i] for i in idxSet])
    return (xOut)

target_url = ("D:/Personal/Desktop/139.csv")
data = csv.reader(open(target_url, encoding='utf-8'))
xList = []
labels = []
names = []
firstLine = True
# for line in data:
#     if firstLine:
#         names = line.strip().split(",")
#         firstLine = False
#     else:
#         row = line.strip().split(",")
#         labels.append(float(row[-1]))
#         row.pop()
#         floatRow = [float(num) for num in row]
#         xList.append(floatRow)
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
    labels.append(float(line[-3]))
    floatline.pop()
    floatline.pop()
    floatline.pop()
    xList.append(floatline)

# print(labels)
# print(xList)
indices = range(len(xList))
xListTest = [xList[i] for i in indices if i%3 == 0]
xListTrain = [xList[i] for i in indices if i%3 != 0]
# print(xListTrain)
labelsTest = [labels[i] for i in indices if i%3 == 0]
# print(labelsTest)
labelsTrain = [labels[i] for i in indices if i%3 != 0]
# print(labelsTrain)
attributeList = []
index = range(len(xList[1])) #12
indexSet = set(index)
# print(indexSet)
indexSeq = []
oosError = []
kkk = 0
for i in index:
    attset = set(attributeList)
    attTrySet = indexSet - attset
    # print(attTrySet)
    #form into list
    attTry = [ii for ii in attTrySet]
    # print(attTry)
    errorList = []
    attTemp = []
    for iTry in attTry:
        attTemp = [] + attributeList
        attTemp.append(iTry)
        # sys.stdout.write("attTemp")
        # print(attTemp)
        xTrainTemp = xattrSelect(xListTrain, attTemp)
        xTestTemp = xattrSelect(xListTest, attTemp)
        # sys.stdout.write("xTrainTemp")
        # print(xTrainTemp)
        # sys.stdout.write("xTestTemp")
        # print(xTestTemp)

        #转为Numpy array
        xTrain = numpy.array(xTrainTemp)
        # print(xTrain)
        yTrain = numpy.array(labelsTrain)
        xTest = numpy.array(xTestTemp)
        yTest = numpy.array(labelsTest)
        fatigueQModel = linear_model.LinearRegression()
        fatigueQModel.fit(xTrain,yTrain)
        rmsError = numpy.linalg.norm((yTest-fatigueQModel.predict(xTest)),2)/sqrt(len(yTest))
        kkk = kkk + 1
        errorList.append(rmsError)
        attTemp = []
    iBest = numpy.argmin(errorList)
    attributeList.append(attTry[iBest])
    oosError.append(errorList[iBest])
print("out of sample error versus attribute set size")
print(oosError)
print("\n" + "Best attribute indices")
print(attributeList)
namesList = [names[i] for i in attributeList]
print("\n" + "Best attribute names")
print(namesList)

#显示与属性数量相对应的error
x = range(len(oosError))
plt.plot(x, oosError, 'k')
plt.xlabel('Numbers of attributes')
plt.ylabel('Error (RMS)')
plt.show()

#构建柱状图，利用得到的模型预测样本外数据
indexBest = oosError.index(min(oosError))
attributeBest = attributeList[1:(indexBest+1)]
xTrainTemp = xattrSelect(xListTrain, attributeBest)
xTestTemp = xattrSelect(xListTest, attributeBest)
xTrain = numpy.array(xTrainTemp)
xTest = numpy.array(xTestTemp)
print("\n" + "训练集")
print(xTrain)
print("\n" + "测试集")
print(xTest)

fatigueQModel = linear_model.LinearRegression()
fatigueQModel.fit(xTrain, yTrain)
errorVector = yTest-fatigueQModel.predict(xTest)
print("\n" + "错误向量")
print(len(errorVector))
print(errorVector)
plt.hist(errorVector)

plt.xlabel("Bin Boundaries")
plt.ylabel("Counts")
plt.show()

plt.scatter(fatigueQModel.predict(xTest),yTest, s=100, alpha=0.10)
plt.xlabel("Predicted fatigue data of blade mx 4")
plt.ylabel('actual fatigue')
plt.show()

print(kkk)
