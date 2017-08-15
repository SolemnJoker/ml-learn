#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
N = 100
K = 3
D = 2
 
def createData():
    X = np.zeros((N * K, D),dtype=float)
    Y = np.zeros(N * K, dtype=float)
    for k in range(K):
        idx = range(N * k, N * (k + 1))
        r = np.linspace(0.0, 1.0, N)
        t = np.linspace(4 * k, (k + 1) * 4, N) + np.random.randn(N) * 0.2
        X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[idx] = float(k)
    print(type(Y),type(Y[0]))
    return X,Y


def train(x,y):
    print(x.shape)
    print(y.shape)
    #labels不是one hot类型标签，用tf.one_hot转换为one_hot标签
    #注意转换后是tensor类型不能作为feed_dict的feed
    y_onehot = tf.one_hot(y,depth=3) 
    print(y_onehot)
    h1_num = 100
    X = tf.placeholder(tf.float32,shape=x.shape,name='x')
    Y = tf.placeholder(tf.float32,shape=y_onehot.shape,name='y')

    keep_prob = tf.placeholder(tf.float32)
    w1 = tf.Variable(tf.truncated_normal([D,h1_num]),dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([1,h1_num]),dtype=tf.float32)
    w2 = tf.Variable(tf.truncated_normal([h1_num,K]),dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([1,K]),dtype=tf.float32)
 
    h1 = tf.nn.relu(tf.matmul(X,w1)+b1)                 #隐层
    h_1_drop = tf.nn.dropout(h1,keep_prob)              #dropout层
    Y_prev = tf.nn.softmax(tf.matmul(h_1_drop,w2)+b2)

    loss = -tf.reduce_sum(Y*tf.log(Y_prev)) # + 1e-6*tf.global_norm([w2,w1])
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
 
    correct_prediction = tf.equal(tf.argmax(Y_prev,1),tf.arg_max(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #y_onehot是tensor类型不能作为feed_dict的feed
        y_r = sess.run(y_onehot)
        for epoch in range(1000):
            sess.run(train_op,feed_dict={X:x,Y:y_r,keep_prob:0.8})
            if epoch % 100 == 0:
                print(epoch,sess.run(accuracy,feed_dict={X:x,Y:y_r,keep_prob:1.0}))
                sys.stdout.flush()
 


        W1,B1,W2,B2 = sess.run([w1,b1,w2,b2])
   
    return W1,B1,W2,B2
    
    


def show(X,Y,W,b,W2,b2):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=10, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.show()

X,Y = createData()
W,b,W2,b2 = train(X,Y)
show(X,Y,W,b,W2,b2)
