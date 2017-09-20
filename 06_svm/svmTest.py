from numpy import *
import matplotlib.pylab as plt
import svmSmoSimple
import svmPlattSmo

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
    y1 = (-1-(weights.item(0)*x + b.item(0))) /weights.item(1)
    y2 = (1-(weights.item(0)*x + b.item(0))) /weights.item(1)
    ax.plot(x,y)
    ax.plot(x,y1,'--')
    ax.plot(x,y2,'--')
    plt.show()

dataMat,labelMat = loadDataSet("./06_svm/testSet.txt")
a1,b1,w1  = svmSmoSimple.smoSimple(dataMat, labelMat, 0.5, 0.01, 10)
plotBestFit(dataMat,labelMat,w1,b1)
a,b,w  = svmPlattSmo.smoPlatt(dataMat, labelMat, 0.5, 0.01, 100)
plotBestFit(dataMat,labelMat,w,b)
print b1,w1
print b,w






