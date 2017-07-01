#-*- coding: UTF-8 -*-
from numpy import *
import matplotlib.pyplot as plt
import os

def loadData():
    f = open(os.path.dirname(__file__)+'/testSet.txt')
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
    dataSet = array(dataSet)
    s=shape(dataSet)
    weights =ones(s[1])
    alpha = 0.01
    for i in range(s[0]):
        h = simgoid((dataSet[i]*weights))
        error = labels[i] - h
        print i,s,dataSet[i]*error,error
        weights = weights + alpha*dataSet[i]*error
    return weights
 
def stocGradAscent1(dataSet,labels,numInter = 150):
    dataSet = array(dataSet)
    s=shape(dataSet)
    weights =ones(s[1])
    for j in range(numInter):      
        dataIndex = range(s[0]) 
        for i in range(s[0]):
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = simgoid(sum(dataSet[randIndex]*weights))
            alpha = 4/(1.0+i+j) + 0.01
            error = labels[randIndex] - h
            weights = weights + alpha*dataSet[randIndex]*error
            print len(dataIndex),randIndex
            del(dataIndex[randIndex])
    return weights



dataset,labels = loadData()
weights = gradAscent(dataset,labels)

weights = stocGradAscent0(dataset,labels)

