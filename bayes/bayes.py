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

def trainNB0(trainMat,classlist):
    numTrainDocs = len(trainMat)
    pAbusive = sum(classlist)/float(numTrainDocs)
    docLen = len(trainMat[0])
    perWordNum0 = ones(docLen)
    perWordNum1 = ones(docLen)
    totalWord0  = 2.0
    totalWord1  = 2.0
    for i in range(numTrainDocs):
        if classlist[i] == 1:
            perWordNum1 += trainMat[i]
            totalWord1  += sum(trainMat[i])
        else:
            perWordNum0 += trainMat[i]
            totalWord0  += sum(trainMat[i])
 
    p1Vec = log(perWordNum1/totalWord1)
    p0Vec = log(perWordNum0/totalWord0)
    return p0Vec,p1Vec,pAbusive

def classifyNB(inputVec,p0Vec,p1Vec,pClass1):
    inputArray =array(inputVec) 
    p1 = sum(inputArray * p1Vec) + pClass1
    p0 = sum(inputArray * p0Vec) + 1 - pClass1
    if p1 > p0:return 1
    else:return 0

def testNB():
    postingList,classlist=loadDataSet()
    vocablist = createVocablist(postingList)
    trainMat = []
    for postDoc in postingList:
        trainMat.append(word2vec(vocablist,postDoc))
    p0Vec,p1Vec,pAbusive = trainNB0(trainMat,classlist)

    testInput = word2vec(vocablist,['love','my','daltation'])
    print testInput,'classify:',classifyNB(testInput,p0Vec,p1Vec,pAbusive)
    testInput = word2vec(vocablist,['stupid','garbege'])
    print testInput,'classify:',classifyNB(testInput,p0Vec,p1Vec,pAbusive)

