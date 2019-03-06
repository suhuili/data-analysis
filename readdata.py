__author__ = 'susu'
import urllib
import numpy as np
import sys
import scipy.stats as stats
import pylab
import pandas as pd
import csv
#读取数据
# target_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
target_url = ("D:/Personal/Desktop/fatigue2.0_2y.csv")
# data = urllib.request.urlopen(target_url)

# data = pd.read_csv(target_url,header=0,encoding="utf-8", prefix="V")/
data = csv.reader(open(target_url, encoding='utf-8'))

#分割数据
xList = []
labels = []
for index,line in enumerate(data):
    #逗号分割
    # row = line.decode().strip().split(',')
    # print(row)
    # xList.append(row)
    if index == 0:
        continue
    xList.append(line)
    # print(xList)
sys.stdout.write("Number of Rows of Data = " + str(len(xList)) + '\n')
sys.stdout.write("Number of Columns of Data = " + str(len(xList[1])) + '\n')
rownum = len( xList )
column = len( xList[1] )

type =  [0] * 3
colCounts = []

for col in range(column):
    for row in xList:
        try:
            a = float(row[col])
            if isinstance(a, float):
                type[0] += 1
        except ValueError:
            if len(row[col]) > 0:
                type[1] += 1
            else:
                type[2] += 1
    colCounts.append(type)
    type = [0] * 3

sys.stdout.write("Col#" + '\t' + "Number" + '\t' +"Strings" + '\t ' + "Other\n")
iCol = 0
for types in colCounts:
    sys.stdout.write(str(iCol) + '\t\t' + str(types[0]) + '\t\t' + str(types[1]) + '\t\t' + str(types[2]) + "\n")
    iCol += 1

#统计某一列的统计信息  Q-Q图 quantile-quantile图
# 0 x, 1 y, 2 z, 3 H, 4 A normal, 5 K normal, 6 平均风速 normal, 7 v50 normal , 8 ve50 normal , 9 湍流强度 normal， 10 空气密度 normal， 11 风切变 normal
# 12 入流角 normal， 13 发电量  ×， 14 叶轮直径， 15 塔筒重量， 16 塔筒图号 ×， 17 叶片型号 ×， 18 叶根mx normal
col = 15
colData = []
for row in xList:
    # if i == 0:
    #     continue
    colData.append(float(row[col]))

stats.probplot(colData, dist="norm", plot=pylab)
pylab.show()
colArray = np.array(colData)
colMean = np.mean(colArray)
colstd = np.std(colArray)
sys.stdout.write("Mean = " + '\t' + str(colMean) + '\t\t' +"Standard Deviation = " + '\t ' + str(colstd) + "\n")

#计算分位数 quantile boundaries
ntiles = 4
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i*(100)/ntiles))

sys.stdout.write("\nBoundaries for 4 Equal Percentiles \n")
print(percentBdry)

sys.stdout.write("\n")

ntiles = 10
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray,i*(100)/ntiles))

sys.stdout.write("\n Boundaries for 10 Equal percentiles \n")
print(percentBdry)
print("\n")

#统计类别变量中的类别
col = 16
colData = []
for row in xList:
    colData.append(row[col])

unique = set(colData)

sys.stdout.write("Unique Lable values \n")
print(unique)

#统计类别变量中类别的个数
catDict = dict(zip(list(unique),range(len(unique))))
print(catDict)
catCount = [0] * 4
for elt in colData:
    catCount[catDict[elt]] += 1

sys.stdout.write("Counts for each value of categorical label \n")
print(list(unique))
print(catCount)