#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os 
import sys

def loaddata(filename):
    data = []
    labels = []
    f = open(filename)
    lines = f.readlines()
    for line in lines:
        lineArray = line.strip().split()
        # z = w*[x,y]+b => w' = [w,b] => z = w'*[x,y,1]多出来的1.0为了相乘时实现+b
        data.append([float(lineArray[0]),float(lineArray[1]),1.0]) 
        labels.append(float(lineArray[-1]))
    return data,labels

def train(data,labels):
    X = tf.placeholder(tf.float32,shape=[len(data),len(data[0])])
    Y = tf.placeholder(tf.float32)
    XT = tf.transpose(X)
    weights = tf.Variable(tf.ones([1,len(data[-1])]),name="weights") #weight 和 b
    
    
    #1.损失函数,交叉熵
    Y_pred = tf.matmul(weights,XT)#注意这里weights和X矩阵相乘X需要转置
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_pred,labels=Y))

    #2.等价于下面的运算,交叉熵原理
    #Y_pred = tf.sigmoid(tf.matmul(weights,XT))
    #loss = -tf.reduce_mean(Y*tf.log(Y_pred) + (1.0-Y)*tf.log(1.0-Y_pred))
 
    #3.也可以这么做,损失函数，平方误差,L2
    #Y_pred = tf.sigmoid(tf.matmul(weights,XT))
    #loss = tf.reduce_mean(tf.square(Y-Y_pred))
 
    #正则化
    loss += 1e-6*tf.global_norm([weights])
    trainOp = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    w = []
    epochs= 3000
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        epoch = 0
        for epoch in range(epochs):
            t,w = sess.run([trainOp,weights],feed_dict={X:data,Y:labels})
            print("step:{},w:{}".format(epoch,w))
    return w


#显示
def plotBestFit(dataset,labels,weights):
    xcord = [[],[]];ycord=[[],[]]
    n = len(dataset)
    for i in range(n):
        xcord[int(labels[i])].append(dataset[i][0])
        ycord[int(labels[i])].append(dataset[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord[0],ycord[0],s = 30,c='red',marker='s')
    ax.scatter(xcord[1],ycord[1],s = 30,c='blue')
    x = np.arange(-3.0,3.0,0.1)
    y = -(weights.item(0)*x + weights.item(2)) /weights.item(1)
    ax.plot(x,y)
    plt.show()

data,labels = loaddata(os.path.dirname(__file__) +"/logistic_regression_test_data.txt")
weights = train(data,labels)
print(weights)
sys.stdout.flush()
plotBestFit(data,labels,weights)



       
