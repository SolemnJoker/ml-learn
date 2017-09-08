## 一.算法流程
### 构建决策树：
**输入**：训练集 $D = \{ (x_1,y_1),(x_2,y_2),...,(x_m,y_m) \}$
　　　 属性集 $A = \{ a_1,a_2,....,a_n \}$
createTree(D,A)
```
if D 中样本全属于同一类C:
    标记当前节点node标记为C类叶子节点
    return C
else if A = 空 OR D中样本在A上取值相同:
    当前节点node标记为叶子节点,类别C为D中样本最多的类
    return C
else
    从A中选择最优属性划分a*
    划分数据集Di
    创建子节点
    for 每个划分的子集:
        createTree(Di,A/a*)
```  
## 二.实现
### 我们的实验数据
    判断一个生物是否是鱼:
    labels = ['no surfacing', 'flippers', 'head']
    dataset = [[0, 1, 1, 'yes'],
               [1, 1, 1, 'yes'],
               [1, 0, 1, 'yes'],
               [1, 1, 0, 'no'],
               [0, 0, 1, 'no'],
               [1, 1, 0, 'no'],
               [0, 0, 0, 'yes']]
 
 
### 寻找最优属性划分
　　算法流程中第8行提到了选择最有属性划分，那么怎么划分最优属性呢。划分属性的原则就是将无序的数据变得更有序。
　　划分数据集之前和之后信息发生的变化称为信息增益，计算每个属性划分数据集的信息增益，信息增益最高的属性就是最好的划分属性。集合的信息度量方式是香农熵，熵(entropy)的定义是信息的期望值。对于根据某个属性做的分类，$x_i$表示其中一个被分类的一类数据则$x_i$的信息期望值为:
  $$l(x_i)=-log_2 p(x_i)$$
  　　$p(x_i)$表示$x_i$出现的概率
　　 所有可能的类别的信息期望值就是熵:
　　$$H = -\sum_{i=1}^n p(x_i)log_2p(x_i)$$
  **计算香农熵代码:**
  这里featVec最后一项是分类，我们用labelcount记录每一个分类的出现个数
  labelcount是一个dict ，使用labelcount.get(curLabel,0)返回当前字典中curLabel的值（这里表示出现次数）,第二个参数0表示如果字典中没有curLabel这个key则插入到字典中，默认值为0
将所有分类出现的次数都记录到labelcount之后就可以遍历labelcount字典用出现次数计算概率,从而计算香农熵
```python
def calShannonEnt(dataset):
    dataSize = len(dataset)
    labelCount = {}
    for featVec in dataset:
        curLabel = featVec[-1]
        labelCount[curLabel] = labelCount.get(curLabel, 0) + 1
        shannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key]) / dataSize
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt
```
 
### 选择最优划分
　　首先计算划分前数据集的信息熵baseEntropy.
　　遍历每一个属性，计算每一个属性划分数据集的信息增益　
```python
  featNum = len(dataset[0]-1)
     for i in range(featNum):
```
 
对于第i个属性划分之后，获取存在的所有属性值,统计这个属性存在的属性值，为了我们的到的属性值唯一，我们使用set来保存属性值
```python
      featList = [X[i] for X in dataset]
      uniqueFeatValue = set(featList)
```
这时我们就可以通过属性值来划分子数据集，遍历属性axis所有可能的属性值， 将属性axis值为value的数据取出作为子数据集的元素
同时计算根据这个属性划分数据集得到的信息增益
注意计算子数据集的香农熵之后还要乘上这个子数据集的概率
```python
       for value in uniqueFeatValue:
           subDataset = splitDataset(dataset, i, value)
           prob = float(len(subDataset)) / len(dataset)
           newEntropy += prob * calShannonEnt(subDataset)
```
 
　**划分数据集：**
将属性axis值为value的数据取出作为子数据集的元素
数据加入到子数据集后，需要把原划分的属性标签去掉
假设我们要得到axis == i的属性,属性值为v的子集
遍历数据集的每一条数据记录 $Data_i = [ x_1,x_2,...,x_i,...x_n]$
当$x_i=v$时，获取x_i前后的元素再连接起来就得到新的数据项${Data_i}_{new} = [ x_1,x_2,...,x_i-1,x_i+1,...x_n]$
使用extend将x_i前后两个list
extend作用是将两个list连接起来
append的作用是向list添加一个元素
```python
axis:需要划分的属性
value:类别
retDataset:返回dataset中属性axis为value的子集
def splitDataset(dataset, axis, value):
    retDataset = []
    for featVec in dataset:
        if featVec[axis] == value:
            retFeatVec = featVec[:axis]
            retFeatVec.extend(featVec[axis + 1:])
            retDataset.append(retFeatVec)
    return retDataset
```
### 构建决策树
构建决策树的函数creatTree是一个递归函数,输入为数据集和列表集,返回的是当前创建的节点,递归返回的条件是:
1. 当前数据集中所有数据都属于同一类
2. 只剩一条数据时
3. 属性集为空
else
先选取最优划分属性，创建节点，节点名字为划分属性，然后在属性集中删除这个属性
使用字典作为数的节点，这样dict的key可以作为当前节点名字，对应的value也用一个dict表示, value的字典保存子节点，这样层层潜逃就可以构成一个树.
勇于保存子节点的dict中，key保存的是当前划分属性的属性值，val为对应的子节点通过递归调用createTree得到
 
获取当前划分属性的所有属性值，用set做唯一储存
```python
    featlist = [f[bestFeat] for f in dataset]
    uniqueFeatValue = set(featlist)
```
对于每一个属性值,属性值作为子节点dict的key，将createTree返回的节点作为val
这里调用splitDataset获得指定属性值的子数据集作为下一层createTree的数据集
```
    for value in uniqueFeatValue:
        subLabels = curLabels[:]
        curTreeNode[bestLabel][value] = createTree(
            splitDataset(dataset, bestFeat, value), subLabels)
 
```
 
### 分类
当都建好决策树后就可以用这个决策树来做分类了
分类函数classify也是一个递归函数，根据输入的属性和属性值从决策树的根节点搜索，直到搜索到叶子节点
我们可以用判断当前节点是不是字典类型来判断当前节点是否是叶子节点,如果是字典类型，则不是叶子节点,不是叶子节点就继续向下搜索
否则返回当前类型
```
def classify(inTree,featLabel,featVec):
    label = inTree.keys()[0]
    featIndex = featLabel.index(label)
    childs = inTree[label]
    nextNode = childs.get(featVec[featIndex],'error')
    if type(nextNode) == type({}):
        result = classify(nextNode,featLabel,featVec)
    else:
        result = nextNode
    return result
```
[构建决策树和分类的完整代码][1]
 
 
  [1]: https://github.com/SolemnJoker/ml-learn/blob/master/02_decision_tree/tree.py