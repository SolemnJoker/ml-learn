from numpy import *
import matplotlib.pylab as plt
from svmSmoSimple import *

def loadDataSet(filename):
    dataset = []
    labels = []
    f = open(filename)
    for line in f.readlines():
        lineArr = line.split('\t')
        dataset.append([float(lineArr[0]),float(lineArr[1])])
        labels.append(float(lineArr[2]))
    return dataset,labels

def plotBestFit(dataset,labels,weights,b):
    xcord = [[],[]];ycord=[[],[]]
    n = shape(dataset)[0]
    for i in range(n):
        xcord[int(labels[i] == 1)].append(dataset[i][0])
        ycord[int(labels[i] == 1)].append(dataset[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord[0],ycord[0],s = 30,c='red',marker='s')
    ax.scatter(xcord[1],ycord[1],s = 30,c='blue')
    x = arange(-1.0,10.0,0.1)
    y = -(weights.item(0)*x + b.item(0)) /weights.item(1)
    ax.plot(x,y)
    plt.show()

dataMat,labelMat = loadDataSet("./06_svm/testSet.txt")
a,b,w  = smoSimple(dataMat, labelMat, 0.5, 0.01, 10)
plotBestFit(dataMat,labelMat,w,b)





