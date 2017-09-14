#-*- coding: UTF-8 -*-
import random
from numpy import *
import matplotlib.pylab as plt

def loadDataSet(filename):
    dataset = []
    labels = []
    f = open(filename)
    for line in f.readlines():
        lineArr = line.split('\t')
        dataset.append([float(lineArr[0]),float(lineArr[1])])
        labels.append(float(lineArr[2]))
    return dataset,labels

def selectJrand(i,m):
    j = i
    while i == j:
        j = random.randint(0,m)
    return j

def calAlphaLH(lableI,labelJ,alphaI,alphaJ,C):
    #label 为1或-1
    if lableI*labelJ < 0:
        L = max(0,alphaJ - alphaI)
        H = min(C,C + alphaJ - alphaI)
    else:
        L = max(0,alphaJ + alphaI - C)
        H = min(C,alphaJ + alphaI)
    return L,H
    
def clipAlpha(a,H,L):
    if a > H:
        a = H
    elif a < L:
        a = L
    return a

def smoSimple(dataset,labels,C,toler,maxIter):
    dataMat = mat(dataset)
    labelsMat = mat(labels).transpose()
    m,n = shape(dataMat)
    alphas = mat(zeros([m,1]))
    iter = 0
    b = 0
    def x(i):
        return dataMat[i,:]
    def K(i,j):
        return x(i)*x(j).T
    while iter < maxIter:
        alphaPairsChange = 0
        for i in range(m):
            # $w = \sum^n_{i=1} a_i y_i x_i$
            weight =  multiply(alphas,labelsMat).T * dataMat 
            fxi = weight*x(i).T + b
            #满足KKT条件：  
            #1.labels[i]*fxi>1  &&  alphas[i]==0  
            #2.labels[i]*fxi==1 &&  0<alphas[i]<C  
            #3.labels[i]*fxi<1  &&  alphas[i]=C  
            Ei =  fxi - float(labels[i]) #定义Ei，则Ei*labels[i]=fxi*labels[i]-labels[i]*labels[i]=fxi*labels[i]-1  
            #根据定义的Ei，可知根据符号 Ei*label[i]与零比较，等价于上面的KKT条件  
            #那么不满足KKT条件的为：  
            #1、Ei*labels[i]>0  &&   alphas[i]>0    需要做优化  
            #2、Ei*labels[i]==0 &&   这个时候数据点i位于边界上，不做优化处理  
            #3、Ei*labels[i]<0  &&   alphas[i]<C    需要做优化  
            if (Ei*labels[i] > toler and alphas[i] > 0) or \
                (Ei *labels[i] < -toler  and alphas[i] < C):
                j = selectJrand(i,m)
                fxj = float(weight*x(j).T) + b
                Ej = fxj - float(labels[j])
                alphaJold = alphas[j]
                alphaIold = alphas[i]
                L,H = calAlphaLH(labels[i],labels[j],alphas[i],alphas[j],C)
                eta = 2*K(i,j) - K(i,i) - K(j,j)
                if eta>=0:print 'eta' ;continue
                alphas[j] -= labels[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                #if abs(alphas[j]  - alphaJold) < 0.00001:print "j not moving enough";continue
                alphas[i] += labels[i]*labels[j]*(alphaJold-alphas[j])
                #计算b，当0<alpha<C时，由kkt条件，y1(wx1+b) =1
                #b = y1 - wx1 ,将w用 \sum_{i=1}^N a_i y_i xi替换,
                #最后简化为b^new = -E_1 - y_1K_{11}(a_1^{new} - a_1^{old}) - y_2K_{21}(a_2^{new} - a_2^{old})
                b1 = -Ei - labels[i]*K(i,i)*(alphas[i] - alphaIold) -\
                 labels[j]*K(j,i)*(alphas[j] - alphaJold) + b

                b2 = -Ej - labels[i]*K(i,j)*(alphas[i] - alphaIold) -\
                labels[j]*K(j,j)*(alphas[j] - alphaJold) + b

                if alphas[i] > 0 and alphas[i] < C:
                    b = b1
                elif alphas[j] > 0 and alphas[j] < C:
                    b = b2
                else:
                    b = (b1+b2)/2
                alphaPairsChange += 1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChange)
        if (alphaPairsChange == 0):
            iter += 1
        else:
            iter = 0
        print "iteration number: %d" % iter
    return alphas,b,weight

dataMat,labelMat = loadDataSet("./06_svm/testSet.txt")
a,b,w  = smoSimple(dataMat, labelMat, 0.5, 0.01, 10)
print b,w

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

plotBestFit(dataMat,labelMat,w,b)






