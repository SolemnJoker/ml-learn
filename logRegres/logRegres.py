from numpy import *
import os

def loadData():
    f = open('testSet.txt')
    lines = f.readlines()
    dataSet = []
    labels  = []
    for line in lines:
        lineArr = line.strip().split()
        dataSet.append([float(lineArr[0]),float(lineArr[1]),1.0])
        labels.append(float(lineArr[2]))
    return dataSet,labels

def simgoid(X):
    return 1.0 / (1.0 + exp(-X))

def gradAscent(dataSet,labels):
    dataMat = mat(dataSet)
    labelMat = mat(labels).transpose()
    s = shape(dataMat)
    weights = ones((s[1],1))
    maxStep = 500
    alpha = 0.001
    for i in range(maxStep):
        h = simgoid(dataMat*weights)
        error = labelMat - h
        weights = weights + alpha*dataMat.transpose()*error
    return weights

def stocGradAscent0(dataSet,labels):
    s=shape(dataSet)
    weights =ones(s[1])
    for i in range(s[0]):
        h = simgoid(dataMat[i]*weights)
        error = labelMat - h
        weights = weights + alpha*dataMat.transpose()*error
    return weights
 
def stocGradAscent1(dataSet,labels):
    s=shape(dataSet)
    weights =ones(s[1])
    for i in range(s[0]):
        h = simgoid(dataMat[i]*weights)
        error = labelMat - h
        weights = weights + alpha*dataMat.transpose()*error
    return weights
 