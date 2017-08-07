#coding:utf-8
#展示如何使用tensorboar scalar绘制训练过程
#控制台:tensorboard --logdir="./my_graph" #启动tensorboard服务，会使用6006端口
#浏览器:http://localhost:6006 #最好用chrome，点击顶部graph看到数据流图
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def createData(W,b):
    X = np.linspace(-1,1,100)
    Y = W*X + b + np.random.randn(*X.shape)*W*0.1
    return X,Y

def train(trainX,trainY):
    #X = tf.placeholder(tf.float32,shape = trainX.shape) 默认shape=None表示任意类型
    #Y = tf.placeholder(tf.float32,shape = trainY.shape)
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    
    W = tf.Variable(np.random.randn(),name='weight')
    B = tf.Variable(np.random.randn(),name='bias')

    Y_pred = W*X + B 
    #也可以Y_pred = tf.add(tf.multiply(W,X) , b)
    
    loss = tf.reduce_sum(tf.square(Y - Y_pred))
    #也可以loss = tf.reduce_sum(tf.squared_difference(Y , Y_pred))

    #正则化
    loss = loss + 1e-6*tf.global_norm([W])

    trainOp = tf.train.GradientDescentOptimizer(0.001).minimize(loss)


    epochs = 1000 
    fig, ax = plt.subplots(1, 1)
    ax.scatter(trainX,trainY)
    fig.show()
    plt.draw()
    with tf.Session() as sess:
        #tensorboard:添加变量到summary中
        tf.summary.scalar("w",W)
        tf.summary.scalar("b",B)
        #tensorboard:merge summary
        merge = tf.summary.merge_all()
        #tensorboard:创建writer
        writer = tf.summary.FileWriter("./linear", sess.graph)
        sess.run(tf.initialize_all_variables())
        epoch = 0
        for epoch in range(epochs):
            t,w,b = sess.run([trainOp,W,B],feed_dict={X:trainX,Y:trainY})
            if epoch % 50 == 0:
                #tensorboard:绘制东西也要sess.run,将返回的Result（summary类型）添加到writer中
                result = sess.run(merge,feed_dict={X:trainX,Y:trainY})
                writer.add_summary(result,epoch)
            print("step:{},w:{},b:{}".format(epoch,w,b))



    print("final:W:{},b:{}".format(w,b))
    writer.close()
    ax.plot(trainX,trainX*w + b)
    plt.show()

X,Y = createData(2,1)
train(X,Y)
        
