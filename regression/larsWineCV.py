__author__ = 'mike-bowles'

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

#Normalize columns in x and labels

nrows = len(xList)
ncols = len(xList[0])

#calculate means and variances
xMeans = []
xSD = []
for i in range(ncols):
    col = [xList[j][i] for j in range(nrows)]
    mean = sum(col)/nrows
    xMeans.append(mean)
    colDiff = [(xList[j][i] - mean) for j in range(nrows)]
    sumSq = sum([colDiff[i] * colDiff[i] for i in range(nrows)])
    stdDev = sqrt(sumSq/nrows)
    xSD.append(stdDev)

#use calculated mean and standard deviation to normalize xList
xNormalized = []
for i in range(nrows):
    rowNormalized = [(xList[i][j] - xMeans[j])/xSD[j] for j in range(ncols)]
    xNormalized.append(rowNormalized)

#Normalize labels
meanLabel = sum(labels)/nrows
sdLabel = sqrt(sum([(labels[i] - meanLabel) * (labels[i] - meanLabel) for i in range(nrows)])/nrows)

labelNormalized = [(labels[i] - meanLabel)/sdLabel for i in range(nrows)]

#Build cross-validation loop to determine best coefficient values.

#number of cross validation folds
nxval = 10

#number of steps and step size
nSteps = 700
stepSize = 0.004

#initialize list for storing errors.
errors = []
for i in range(nSteps):
    b = []
    errors.append(b)


for ixval in range(nxval):
    #Define test and training index sets
    idxTest = [a for a in range(nrows) if a%nxval == ixval]
    idxTrain = [a for a in range(nrows) if a%nxval != ixval]

    #Define test and training attribute and label sets
    xTrain = [xNormalized[r] for r in idxTrain]
    xTest = [xNormalized[r] for r in idxTest]
    labelTrain = [labelNormalized[r] for r in idxTrain]
    labelTest = [labelNormalized[r] for r in idxTest]

    #Train LARS regression on Training Data
    nrowsTrain = len(idxTrain)
    nrowsTest = len(idxTest)

    #initialize a vector of coefficients beta
    beta = [0.0] * ncols

    #initialize matrix of betas at each step
    betaMat = []
    betaMat.append(list(beta))

    for iStep in range(nSteps):
        #calculate residuals
        residuals = [0.0] * nrows
        for j in range(nrowsTrain):
            labelsHat = sum([xTrain[j][k] * beta[k] for k in range(ncols)])
            residuals[j] = labelTrain[j] - labelsHat

        #calculate correlation between attribute columns from normalized wine and residual
        corr = [0.0] * ncols

        for j in range(ncols):
            corr[j] = sum([xTrain[k][j] * residuals[k] for k in range(nrowsTrain)]) / nrowsTrain

        iStar = 0
        corrStar = corr[0]

        for j in range(1, (ncols)):
            if abs(corrStar) < abs(corr[j]):
                iStar = j; corrStar = corr[j]

        beta[iStar] += stepSize * corrStar / abs(corrStar)
        betaMat.append(list(beta))

        #Use beta just calculated to predict and accumulate out of sample error - not being used in the calc of beta
        for j in range(nrowsTest):
            labelsHat = sum([xTest[j][k] * beta[k] for k in range(ncols)])
            err = labelTest[j] - labelsHat
            errors[iStep].append(err)
            # print(errors[iStep])

cvCurve = []
print(len(errors))
print(len(errors[0]))

for errVect in errors:
    mse = sum([x*x for x in errVect])/len(errVect)
    cvCurve.append(mse)

print(cvCurve)
print(len(cvCurve))

minMse = min(cvCurve)
minPt = [i for i in range(len(cvCurve)) if cvCurve[i] == minMse ][0]
print("Minimum Mean Square Error", minMse)
print("Index of Minimum Mean Square Error", minPt)

xaxis = range(len(cvCurve))
plot.plot(xaxis, cvCurve)

plot.xlabel("Steps Taken")
plot.ylabel(("Mean Square Error"))
plot.show()
