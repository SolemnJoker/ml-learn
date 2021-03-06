#-*- coding: UTF-8 -*-
from math import log
import operator
import pickle

# 计算香农熵


def calShannonEnt(dataset):
    dataSize = len(dataset)
    labelCount = {}
    for featVec in dataset:
        curLabel = featVec[-1]
        labelCount[curLabel] = labelCount.get(curLabel, 0) + 1
    shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key]) / dataSize
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataset():
    dataset = [[0, 1, 1, 'yes'],
               [1, 1, 1, 'yes'],
               [1, 0, 1, 'yes'],
               [1, 1, 0, 'no'],
               [0, 0, 1, 'no'],
               [1, 1, 0, 'no'],
               [0, 0, 0, 'yes']]
    labels = ['no surfacing', 'flippers', 'head']
    return dataset, labels

'''
划分子数据集
axis:需要划分的属性
value:类别
retDataset:返回dataset中属性axis为value的子集
'''
def splitDataset(dataset, axis, value):
    retDataset = []
    for featVec in dataset:
        if featVec[axis] == value:
            retFeatVec = featVec[:axis]
            retFeatVec.extend(featVec[axis + 1:])
            retDataset.append(retFeatVec)
    return retDataset

# 选择最佳划分


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
            subDataset = splitDataset(dataset, i, value)
            prob = float(len(subDataset)) / len(dataset)
            newEntropy += prob * calShannonEnt(subDataset)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeat = i

    return bestFeat

# 选择classlist中数量最多的类别


def majorityCnt(classlist):
    classCount = {}
    for c in classlist:
        classCount[c] = classCount.get(c, 0) + 1

    sortClassCount = sorted(classCount.iteritems(),
                            key=operator.itemgetter(1), reverse=True)
    return sortClassCount[0][0]


# 构建决策树
def createTree(dataset, labels):
    curLabels = labels[:]
    classlist = [c[-1] for c in dataset]
    #所有节点都属于同一类
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    #只剩一个数据
    if len(dataset[0]) == 1:
        return majorityCnt(classlist)
    #属性集为空
    if len(labels) == 0:
        return majorityCnt(classlist)
    bestFeat = chooseBestFeat2Split(dataset)
    bestLabel = curLabels[bestFeat]
    curTreeNode = {bestLabel: {}}
    del(curLabels[bestFeat])
    featlist = [f[bestFeat] for f in dataset]
    uniqueFeatValue = set(featlist)
    for value in uniqueFeatValue:
        subLabels = curLabels[:]
        curTreeNode[bestLabel][value] = createTree(
            splitDataset(dataset, bestFeat, value), subLabels)
    return curTreeNode

#分类
def classify(inTree,featLabel,featVec):
    label = inTree.keys()[0]
    featIndex = featLabel.index(label)
    childs = inTree[label]
    nextNode = childs.get(featVec[featIndex],'error')
    if type(nextNode) == type({}):
        result = classify(nextNode,featLabel,featVec) 
    else:
        result = nextNode
    return result

def storeTree(filename,inTree):
    f = open(filename,'w')
    pickle.dump(inTree,f)
    f.close()

def loadTree(filename):
    return pickle.loads(filename)

'''
def createAtree():
    dataset, labels = createDataset()
    return createTree(dataset, labels)
'''