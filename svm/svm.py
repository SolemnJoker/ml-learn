#-*- coding: UTF-8 -*-
import random
from numpy import *

def loadDataset(filename):
    dataset = []
    labels = []
    f = open(filename)
    for line in f.readlines():
        lineArr = line.split('\t')
        dataset.append([float(lineArr[0])),float(lineArr[1])])
        labels.append(float(lineArr[2]))
    return dataset,labels

def selectJrand(i,m):
    j = i
    while i == j:
        j = random.randint(0,m)
    return j

def clipAlpha(a,H,J):
    if a > H:
        a = H
    else if a < L:
        a = L
    return a

def smoSimple(dataset,labels,C,toler,maxIter):
    dataMat = mat(dataset)
    labelsMat = mat(labels).transpose()
    m,n = shape(dataMat)
    alphas = mat(zeros(m,1))
    iter = 0
    b = 0
    while iter < maxIter:
        alphaPairsChange = 0
        Fxi = float(multiply(alphas,labelsMat).T * (dataMat*dataMat[i,:].T) ) + b