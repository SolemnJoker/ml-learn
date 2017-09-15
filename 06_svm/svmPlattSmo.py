#-*- coding: UTF-8 -*-
import random
from numpy import *
class optStruct:
    def __init__(self,dataMat,classLabels,C,toler):
        self.X = dataMat
        self.labels = classLabels
        self.C = C
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros([self.m,1]))
        self.b = 0
        self.eCache = zeros(self.m,2)#第一个元素用于记录Ek是否有效
        self.weights

    def x(self,i):
        return self.dataMat[i,:]
    def K(self,i,j):
        return self.x(i)*self.x(j).T
 

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

def calcEk(oS,k):
    fxk = oS.weight*oS.x(k).T + b
    Ek =  fxk - float(oS.labels[k])
    return Ek

def updateWeight(oS,i):
    oS.weight =  multiply(oS.alphas,oS.labels).T * oS.dataMat 

def updateEk(oS,k):
    oS.eCache[k] = [1,calcEk(oS,k)]

def selectJE(oS,i,Ei):
    maxJ = -1
    maxDeltaE = 0
    Ej = 0
    validECacheList = nonzero(oS.eCacha[:,0])
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
def innerLK(oS,i):
    pass


def smoPlatt(dataset,labels,C,toler,maxIter):
    pass