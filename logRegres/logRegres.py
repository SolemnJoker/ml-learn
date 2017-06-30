from numpy import *
import os
import matplotlib.pylab as plt

def loadData():
    f = open(os.path.dirname(__file__) + '/testSet.txt')
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

def plotBestFit(dataset,labels,weights):
    xcord = [[],[]];ycord=[[],[]]
    n = shape(dataset)[0]
    for i in range(n):
        xcord[int(labels[i])].append(dataset[i][0])
        ycord[int(labels[i])].append(dataset[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord[0],ycord[0],s = 30,c='red',marker='s')
    ax.scatter(xcord[1],ycord[1],s = 30,c='blue')
    x = arange(-3.0,3.0,0.1)
    y = -(weights.item(0)*x + weights.item(2)) /weights.item(1)
    ax.plot(x,y)
    plt.show()

dataSet,labels = loadData()
weights = gradAscent(dataSet,labels)
plotBestFit(dataSet,labels,weights)

        