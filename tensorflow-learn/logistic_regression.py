#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os 

def loaddata(filename):
    data = []
    labels = []
    f = open(filename)
    lines = f.readlines()
    for line in lines:
        lineArray = line.strip().split()
        data.append([float(lineArray[0]),float(lineArray[1])]) #y = w*x+b 多出来的1.0为了相乘时实现+b
        labels.append(float(lineArray[-1]))
    return data,labels

def train(data,labels):
    X = tf.placeholder(tf.float32,shape=[len(data),len(data[0])])
    Y = tf.placeholder(tf.float32)
    XT = tf.transpose(X)
    weights = tf.Variable(tf.ones([1,len(data[-1])]),name="weights") #weight 和 b
    
    Y_pred = tf.matmul(weights,XT)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_pred,labels=Y))
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
    y = -(weights.item(0)*x +1) /weights.item(1)
    ax.plot(x,y)
    plt.show()

print 
data,labels = loaddata(os.path.dirname(__file__) +"/logistic_regression_test_data.txt")
weights = train(data,labels)
plotBestFit(data,labels,weights)



       
