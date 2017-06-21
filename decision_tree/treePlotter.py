import matplotlib.pyplot as plt

def getNumOfLeafs(curTreeNode):
    numLeafs = 0
    childNodes = curTreeNode.values()
    for key in childNodes.keys():
        if type(childNodes[key]) == type({}):
            numLeafs += getNumOfLeafs(childNodes[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(curTreeNode):
    maxDepth = 0
    childNodes = curTreeNode.values()
    curDepth = 1
    for key in childNodes.keys():
        if type(childNodes[key]) == type({}):
            curDepth += getTreeDepth(childNodes[key])
    if curDepth > maxDepth: maxDepth = curDepth
    return maxDepth

