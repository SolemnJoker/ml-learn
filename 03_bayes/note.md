##  预备知识
### 条件概率
考虑这么一个问题有两个箱子A,B,
A中有2个红球，2个白球 A=[r,r,w,w]
B中有3个红球，2个白球 B=[r,r,r,w,w]
总共有4个红球5个白球
**问题1:** 如果从这9个球中取一个球取到红球，白球概率分别为多少？
　　显然:$p(r) = \frac{5}{9}$, $p(w) = \frac{4}{9}$
 
**问题2:** 如果已知球取自A箱取到红球概率为多少？
　　显然:$p(r|A) = \frac{2}{4}$
　　我们还知道 $p(r \; and \;A) =p(A) \times p(r|A)=\frac{4}{9} \times \frac{2}{4}=\frac{2}{9}$
 
**问题3:** 如果已知拿到的是白球，那么这个求是来自A箱的概率是多少呢？
　　A中有三个白球，B中有1个白球，共有4个白球, 那么显然这个球从A中取的概率是$p(A|w)=\frac{2}{4}$, 我们看看来自A并且是白球的概率:
$$p(w \; and \;A)  =p(A) \times p(w|A)=\frac{4}{9} \times \frac{2}{4}\\
= p(w) \times p(A|w)=\frac{4}{9} \times \frac{2}{4} = \frac{2}{9}\\
  $$
 
根据上式可以得出:
$$p(A|w) = \frac{p(w|A) \times p(A)}{p(w)}$$
这就是贝叶斯准则。
 
## 文本分类问题
　　 以在线社区留言为例，下面是某社区留言本的留言统计,作为训练数据,postingList是留言记录，每一行是一条留言，classVec记录对应的分类，0表示非侮辱性留言$c_0$，1表示侮辱性留言$c_1$
```python
     postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
```
**目标：**
　　当输入一条新的留言的时候判断是否是侮辱性留言,设$w$为输入的留言
　　那么问题就是$p(c_i|w) = \frac{p(w|c_i) \times p(c_i)}{p(w)}$，计算出每一个$p(c_i|w)$后只需要找出概率最大的$p(c_i|w)$就可以确定类别了， 由于是比较大小所以$p(w)$相同的情况下只需要计算$p(c_i|w) = p(w|c_i) \times p(c_i)$用于比较即可. 对于训练集数据，$p(w)$表示单词在所有训练文本中出现的概率，$p(c_i)$表示训练集中某一个类别出现的概率，$p(w|c_i)$
 
**词向量:**
　　由于句子和单词用来计算或者用来统计比较麻烦,所以我们可以把训练集中的单词转为词向量来方便统计。词向量$w$是一个元素为0或1的list，这个list对应训练集中出现的所有单词，每一个元素对应一个单词，0代表没有出现，1代表出现。
  1.生成单词表
　　首先需要一个包含所有单词的单词表vocablist,单词表里不需要用一个单词重复出现，因此对于训练集输入的所有句子，先用set保存单词再转换成list
```python
def createVocablist(dataset):
    vocablist = set()
    for doc in dataset:
        vocablist = vocablist | set(doc)
    return list(vocablist)
```
2.生成词向量
　　inputSet为当前输入的句子，返回的词向量为句子中的单词在单词表中的出现情况，先构建一个值为0长度和vocablist一样的词向量，遍历输入的句子inputSet,中的所有单词，如果单词是单词表里的就像对应位置的词向量元素置为1。这里输入的句子中有可能会出现单词表中不存在的单词，这里我们可以忽略这些单词。
```python
def word2vec(vocablist,inputSet):
    ret = [0]*len(vocablist)
    for word in inputSet:
        if word in vocablist:
            ret[vocablist.index(word)] = 1
        else:
            pass
    return ret
```
**处理训练集:**
　　首先需要将训练集中的原始数据转换为词向量的形式，对于每一个留言生成称为一个词向量
　　处理后的训练集trainMat = [ [0,1,0,0,1....1,0,1],
　　　　　　　　　　　　　　  [1,1,0,0,1....1,1,1],
　　　　　　　　　　　　　　  ...
　　　　　　　　　　　　　　  [0,1,0,1,0....1,0,1],
　　　　　　　　　　　　　　  [0,0,0,0,1....0,0,1], ]
　　形式，每一个词向量的长度和vocavlist长度一致。
 
**训练数据:**
　　训练数据就是统计$p(c_i)$ 和$p(w|c_i)$,
 　　对于$p(c_i)$，classlist为0,1向量形式，只需要    pAbusive = sum(classlist)/float(len(classlist)),即可得到$p(c_1)$ ,$p(c_0) = 1-p(c_1)$
　　对于$p(w|c_i) = \frac{每一个单词出现次数}{所有单词出现次数} |c_i$, 由于可能这里除法出现极小的数，对计算不利，我们可以用log来转换一下,后面的乘法也可以改为加法计算  
　　训练结束后我们得到了$p(c_1)$和$p(c_0) = 1-p(c_1)$,以及$p(w|c_i)$,(p1Vec,p2Vec)注意这里的w是一个向量($w=[w_1,w_2...,w_n]$),表示每一个词在$c_i$中出现的概率
```python
    for i in range(numTrainDocs):
        if classlist[i] == 1:
            perWordNum1 += trainMat[i]
            totalWord1  += sum(trainMat[i])
        else:
            perWordNum0 += trainMat[i]
            totalWord0  += sum(trainMat[i])
 
    p1Vec = log(perWordNum1/totalWord1)
    p0Vec = log(perWordNum0/totalWord0)
```
**分类:**
　当有新的句子需要分类时，需要将句子先转换为词向量　
 
```python
    testInput = word2vec(vocablist,['love','my','daltation'])
    testInput = word2vec(vocablist,['stupid','garbege'])
```
  然后将输入的词向量和p1Vec，p0Vec跟别相乘判断拿一个类别概率更大即可
```python
  def classifyNB(inputVec,p0Vec,p1Vec,pClass1):
    inputArray =array(inputVec)
    p1 = sum(inputArray * p1Vec) + pClass1
    p0 = sum(inputArray * p0Vec) + 1 - pClass1
    if p1 > p0:return 1
    else:return 0
```
[完整代码][1]
 
 
 
 
 
 
 
  [1]: https://github.com/SolemnJoker/ml-learn/tree/master/03_bayes