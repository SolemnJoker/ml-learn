#-*- coding: UTF-8 -*-
from math import log

#计算香农熵
def calShannonEnt(dataset):
    dataSize = len(dataset)
    labelCount = {}
    for featVec in dataset:
        curLabel = featVec[-1]
        labelCount[curLabel] = labelCount.get(curLabel,0) + 1
        shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key])/dataSize
        shannonEnt -= prob* log(prob,2)
    return shannonEnt

def createDataset():
    dataset = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [1,0,'no'],
               [1,1,'yes']]
    labels = ['no surfacing','flippers']
    return dataset,labels

#划分子数据集
def splitDataset(dataset,axis,value):
    retDataset = []
    for featVec in dataset:
        if featVec[axis] == value:
            retFeatVec = featVec[:axis]
            retFeatVec.extend(featVec[axis+1:])
            retDataset.append(retFeatVec)
    return retDataset

def chooseBestFeat2Split(dataset):
    bestFeat = -1
    bestInfoGain = 0.0
    featNum = len(dataset[0]) - 1
    baseEntropy = calShannonEnt(dataset)
    for i in range(featNum):
        featList = [X[i] for X in dataset]
        uniqueFeatValue = set(featList)
        newEntropy = 0.0
        for value in uniqueFeatValue:
            subDataset = splitDataset(dataset,i,value)
            prob = float(len(subDataset))/len(dataset)
            newEntropy += prob*calShannonEnt(subDataset)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeat = i
    return bestFeat

#test
dataset,labels = createDataset()
print chooseBestFeat2Split(dataset)

