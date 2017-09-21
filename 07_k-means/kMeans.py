from numpy import *
import matplotlib.pyplot as plt
def showResult(dataset,centrids,clusterAssment):
    m,n = shape(centrids)
    xcord = [[] for i in range(m)];ycord=[[] for i in range(m)]
    num = shape(dataset)[0]
    for i in range(num):
        xcord[int(clusterAssment[i][0])].append(dataset[i][0])
        ycord[int(clusterAssment[i][0])].append(dataset[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ca = ['m','g','b','y']
    sa = ['s','x','d','<']
    for i in range(m):
        ax.scatter(xcord[i],ycord[i],c=ca[i],s = 30)
    ax.scatter(centrids[:,0],centrids[:,1],s = 40,c=ca,marker='x')
    plt.show()


def loadDataSet(filename):
    f = open(filename)
    lines = f.readlines()
    dataSet = []
    for line in lines:
        lineArray = line.strip().split()
        dataSet.append(map(float,lineArray))
    return dataSet

def distL2(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataSet,k):
    n = shape(dataSet)[1]
    centroids = zeros([k,n])
    for i in range(n):
        minI = min(dataSet[:,i])
        maxI = max(dataSet[:,i])
        rangeI = maxI - minI
        centroids[:,i] = minI + rangeI*random.rand(k)
    return centroids

def kMeans(dataSet,k):
    dataSet = array(dataSet)
    m = shape(dataSet)[0]
    clusterAssment = zeros([m,1])
    centroidIsChange = True
    centroids = randCent(dataSet,k)
    while centroidIsChange:
        centroidIsChange = False
        for i in range(m):
            minDist = inf
            minIndex = 0
            for j in range(k):
                dist = distL2(dataSet[i,:],centroids[j,:])
                if dist < minDist:
                    minDist = dist
                    minIndex = j

            if minIndex != clusterAssment[i]:
                centroidIsChange = True
            clusterAssment[i] = minIndex
        print centroids
        for cent in range(k):
            ptsInCluters = dataSet[(nonzero(clusterAssment == cent)[0])]
            centroids[cent]= mean(ptsInCluters,axis=0)
        showResult(dataSet,centroids,clusterAssment)
    return centroids,clusterAssment

dataSet = loadDataSet("./07_k-means/testSet.txt")
centroids,clusterAssment = kMeans(dataSet,4)
