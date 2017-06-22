import matplotlib.pyplot as plt
import tree as t

def getNumOfLeafs(curTreeNode):
    numLeafs = 1
    childNodes = curTreeNode.values()[0]
    for key in childNodes.keys():
        if type(childNodes[key]) == type({}):
           numLeafs += getNumOfLeafs(childNodes[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(curTreeNode):
    maxDepth = 0
    childNodes = curTreeNode.values()[0]
    curDepth = 1
    for key in childNodes.keys():
        if type(childNodes[key]) == type({}):
            curDepth += getTreeDepth(childNodes[key])
    if curDepth > maxDepth:
        maxDepth = curDepth
    return maxDepth

decisionNode = dict(boxstyle='sawtooth',fc='0.8')
leafNode = dict(boxstyle='round4',fc='0.8')
arrow_args = dict(arrowstyle="<-")
def plotNode(nodeText,childPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeText,xy=parentPt,xycoords='axes fraction',xytext=childPt,
    textcoords='axes fraction',va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)

def plotMidText(childPos, parentPos, textStr):
    midX = (childPos[0] + parentPos[0]) / 2.0
    midY = (childPos[1] + parentPos[1]) / 2.0
    createPlot.ax1.text(midX, midY, textStr, va='center',ha='center')


def plotTree(curTreeNode, leftTopPos,parentPos, nodeText):
    numLeafs = getNumOfLeafs(curTreeNode)
    depth = getTreeDepth(curTreeNode)
    firstStr = curTreeNode.keys()[0]
    childs = curTreeNode[firstStr]
    numList=[]
    curIndex = 0
    i = 0
    childslen = len(childs)
    for key in childs.keys():
        if type(childs[key]) == type({}):
            numList.append(getNumOfLeafs(childs[key]))
            if i < childslen/2 :
                curIndex +=numList[-1] 
        else:
            numList.append(1)
            if i < childslen/2 :
                curIndex += 1
        i+=1

    if childslen % 2 ==0:
        pass
    else:
        curIndex = sum(numList)/2 + 1
    
    
 
    curPos = (leftTopPos[0]  + curIndex/plotTree.totalW,leftTopPos[1])
    firstChildPos = (leftTopPos[0] ,leftTopPos[1] - 1/plotTree.totalD)
    curChildX,curChildY = firstChildPos
    if leftTopPos == parentPos:
        parentPos = curPos
    plotNode(firstStr,curPos,parentPos,decisionNode)
    plotMidText(curPos,parentPos,nodeText)
    i = 0
    for key in childs.keys():
        if i == len(childs)/2:
            curChildX += 1.0/plotTree.totalW
        i += 1
        if type(childs[key]) == type({}):
            childNumLeafs,childDepth = plotTree(childs[key],(curChildX,curChildY),curPos,str(key))
            curChildX += childNumLeafs/plotTree.totalW
        else:
            plotNode(childs[key],(curChildX,curChildY),curPos,leafNode)
            plotMidText((curChildX,curChildY),curPos,str(key))
            curChildX += 1.0/plotTree.totalW
    
    return numLeafs,depth



def createPlot(inTree):
    fig = plt.figure(1, facecolor='#ffffff')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumOfLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree(inTree, (0.0, 1.0),(0.0,1.0), '')
    plt.show()

tree = t.createAtree();
print tree
createPlot(tree)