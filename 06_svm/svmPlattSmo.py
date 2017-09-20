#-*- coding: UTF-8 -*-
import random
from numpy import *
import sys
class optStruct:
    def __init__(self,dataMat,classLabels,C,toler):
        self.X = dataMat
        self.labels = classLabels
        self.labelsMat = mat(classLabels).transpose()
        self.C = C
        self.m = shape(dataMat)[0]
        self.alphas = zeros([self.m,1])
        self.alphasMat = mat(zeros([self.m,1]))
        self.b = 0
        self.eCache = zeros([self.m,2])#第一个元素用于记录Ek是否有效
        self.weight = []
        self.toler = toler

    def x(self,i):
        return self.X[i,:]
    def K(self,i,j):
        return self.x(i)*self.x(j).T
 

def selectJrand(i,m):
    j = i
    while i == j:
        j = int(random.uniform(0,m))
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

def calcEk(oS,k):
    fxk = float(oS.weight*oS.x(k).T + oS.b)
    Ek =  fxk - float(oS.labels[k])
    return Ek

def updateWeight(oS):
    oS.weight =  multiply(oS.alphas,oS.labelsMat).T * oS.X 

def updateEk(oS,k):
    oS.eCache[k] = [1,calcEk(oS,k)]

def selectJE(oS,i,Ei):
    oS.eCache[i] = [1,Ei]
    maxJ = -1
    maxDeltaE = 0
    Ej = 0
    validECacheList = nonzero(oS.eCache[:,0])
    if (len(validECacheList)> 1):
        for k in validECacheList:
            if k != i:
                Ek = calcEk(oS,k)
                deltaE = abs(Ek - Ei)
                if deltaE > maxDeltaE:
                    maxDeltaE = deltaE
                    maxJ = k
                    Ej = Ek
        return maxJ,Ej
    else:
        j = selectJrand(i,oS.m)
        Ej = calcEk(oS,j)
        return j,Ej

def innerLK(i,oS):
    updateWeight(oS)
    Ei = calcEk(oS,i)
    if (Ei*oS.labels[i] > oS.toler and oS.alphas[i] > 0) or \
        (Ei *oS.labels[i] < -oS.toler  and oS.alphas[i] < oS.C):
        j,Ej = selectJE(oS,i,Ei)
        alphaJold = oS.alphas[j].copy()
        alphaIold = oS.alphas[i].copy()
        L,H = calAlphaLH(oS.labels[i],oS.labels[j],oS.alphas[i],oS.alphas[j],oS.C)
        eta = float(2*oS.K(i,j) - oS.K(i,i) - oS.K(j,j))
        if eta>=0:
            print 'eta' 
            return 0
        oS.alphas[j] -= oS.labels[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        oS.alphas[i] += oS.labels[i]*oS.labels[j]*(alphaJold-oS.alphas[j])
        updateEk(oS,j)
        updateEk(oS,i)
        b1 = -Ei - oS.labels[i]*oS.K(i,i)*(oS.alphas[i] - alphaIold) -\
        oS.labels[j]* oS.K(j,i) *(oS.alphas[j] - alphaJold) + oS.b

        b2 = -Ej - oS.labels[i]*oS.K(i,j)*(oS.alphas[i] - alphaIold) -\
        oS.labels[j]* oS.K(j,j) *(oS.alphas[j] - alphaJold) + oS.b

        if oS.alphas[i] > 0 and oS.alphas[i] < oS.C:
            oS.b = b1
        elif oS.alphas[j] > 0 and oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1+b2)/2.0
        return 1
    return 0
        
 



def smoPlatt(dataset,labels,C,toler,maxIter):
    oS = optStruct(mat(dataset),labels,C,toler)
    enterSet = True
    alphaPairsChange = 0
    iter = 0
    while((iter < maxIter and alphaPairsChange > 0) or enterSet ):
        alphaPairsChange = 0
        if enterSet:
            for i in range(oS.m):
                alphaPairsChange += innerLK(i,oS)
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChange)
            iter +=1
        else:
            nonBoundIs = nonzero((oS.alphas > 0) * (oS.alphas < C))[0]
            print nonBoundIs
            for i in nonBoundIs:
                alphaPairsChange += innerLK(i,oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChange)
            iter +=1
        if enterSet:enterSet = False
        elif alphaPairsChange == 0:enterSet = True
    print "iteration number: %d" % iter
    updateWeight(oS)
    return oS.alphas,oS.b,oS.weight