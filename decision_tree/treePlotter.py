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

def plotMidText(childPos,parentPos,textStr):
    midX = (childPos[0] + parentPos[0])/2.0
    midY = (childPos[1] + parentPos[1])/2.0
    createPlot.ax1.text(midX,midY,textStr,va='center',ha='center',rotation=90)

def plotTree(curTreeNode,parentPos,nodeText):
    numLeafs = getNumOfLeafs(curTreeNode)
    depth = getTreeDepth(curTreeNode)
    firstStr=curTreeNode.keys()[0]
    
def createPlot(inTree):
    fig = plt.figure(1,facecolor='0xffffff')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getNumOfLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plottree.totalW
    plottree.yOff = 1.0 
    plottree(inTree,(0.5,1.0),'')
    plt.show()
