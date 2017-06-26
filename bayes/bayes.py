from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
 
def createVocablist(dataset):
    vocablist = set()
    for doc in dataset:
        vocablist = vocablist | set(doc)
    return list(vocablist)

def word2vec(vocablist,inputSet):
    ret = [0]*len(vocablist)
    for word in inputSet:
        if word in vocablist:
            ret[vocablist.index(word)] = 1
        else:
            pass
    return ret

def createTrainMat(postinglist):
    vocablist = createVocablist(postinglist)
    trainMat = []
    for postDoc in postinglist:
        trainMat.append(word2vec(vocablist,postDoc))
    return trainMat

def trainNB0(trainMat,classlist):
    numTrainDocs = len(trainMat)
    pAbusive = sum(classlist)/float(numTrainDocs)
    docLen = len(trainMat[0])
    perWordNum0 = zeros(docLen)
    perWordNum1 = zeros(docLen)
    totalWord0  = 0.0
    totalWord1  = 0.0
    for i in range(numTrainDocs):
        if classlist[i] == 1:
            perWordNum1 += trainMat[i]
            totalWord1  += sum(trainMat[i])
        else:
            perWordNum0 += trainMat[i]
            totalWord0  += sum(trainMat[i])
 
    p1Vec = perWordNum1/totalWord1
    p0Vec = perWordNum0/totalWord0
    return p0Vec,p1Vec,pAbusive

