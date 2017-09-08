## 一、knn简介
　　k临近算法采用测量不同特征值之间的距离来分类,在样本数据及中找出k个待分类数据最相似的样本，这k个样本中出现最多的类别作为待分类样本的类别
### 算法流程
```
遍历所有样本
    对输入数据计算和每一个样本数据的误差
    找出k个误差最小的样本
    统计k个最相似的样本中出现最多的类别作为待分类样本类别
```
## 二、代码实现
### 样本数据
这是一份约会网站数据,
**数据有三种分类：**
1. largeDoses 极具魅力的人
2. smallDoses 魅力一般的人
 3. didntlike      不喜欢的人
 
**每条数据共有三种特征:**
1. 每年乘飞机飞行里程数
2. 玩游戏时间百分比
3. 每周消耗冰欺凌公升数
 
```
40920    8.326976    0.953952    largeDoses
14488    7.153469    1.673904    smallDoses
26052    1.441871    0.805124    didntLike
75136    13.147394    0.428964    didntLike
38344    1.669788    0.134296    didntLike
72993    10.141740    1.032955    didntLike
35948    6.830792    1.213192    largeDoses
42666    13.276369    0.543880    largeDoses
67497    8.631577    0.749278    didntLike
35483    12.273169    1.508053    largeDoses
... ...
```
这些数据保存在datingTestData.txt中。
 
### 读取数据
将数据文件中特征值读取为numpy 3*n数组ret_mat
标签去读取为1*n数组labels
```
    for line in lines:
        line = line.strip()
        l = line.split('\t')
        ret_mat[index] = l[0:3]
        labels.append(l[-1])
        index += 1
```
 
 
----------
 
 
### 数据归一化
　　不同评价指标往往具有不同的量纲和量纲单位，这样的情况会影响到数据分析的结果，为了消除指标之间的量纲影响，需要进行数据标准化处理，以解决数据指标之间的可比性。原始数据经过数据标准化处理后，各指标处于同一数量级，适合进行综合对比评价.
  #### 常用的归一化方法有:
  * min-max标准化法
  * Z-score标准化方法
  这里我们使用min-max标准化法。
  $\frac{x-minval}{maxval-minval}$
 
  ### 分类
　　 knn与其他机器学习方法不同的是，没有训练过程，直接对数据集上所有数据计算。分类时需要计算待分类样本与每一个样本数据的相似度，这里相似度我们用L2距离(就是我们常见的欧氏距离)表示
$L_2 = \sqrt{\sum_{k=1}^n x_{ik} - x_{jk}}$
简单的说就是向量每个元素平方和再开根，距离越小误差越小，即相似度越大
```
def classify0(in_x,dataset,labels,k):
    dataset_size = dataset.shape[0]
    diff_mat = in_x - dataset # numpy Broadcasting
    sq_mat = diff_mat**2
    sq_distance = sq_mat.sum(axis=1)
    distances = sq_distance**0.5
    sort_distance_index = distances.argsort()
    class_count = {}
 
    for i in range(k):
        cur_label = labels[sort_distance_index[i]]
        class_count[cur_label] = class_count.get(cur_label,0) + 1
        sort_class_cout = sorted(class_count.iteritems(),key = operator.itemgetter(1),reverse=True)
 
    return sort_class_cout[0][0]
 
```
 
###测试我们的分类器
　　文本中有1000条数据，我们将其中一部分用来做测试数据，一部分用来做样本库，每隔1/10 取一条数据作为测试数据
```
    test_data = dataset[0:data_size/10:1,0:]
    test_labels = labels[0:data_size/10:1]
 
```
剩下的作为样本库
```
    dataset = dataset[data_size/10:data_size:1,0:]
    labels = labels[data_size/10:data_size:1]
```
 
 
[完整代码][1]
 
  [1]: https://github.com/SolemnJoker/ml-learn/tree/master/01_knn