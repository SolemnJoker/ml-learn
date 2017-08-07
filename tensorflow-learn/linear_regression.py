import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def createData(W,b):
    X = np.linspace(-1,1,100)
    Y = W*X + b + np.random.randn(*X.shape)*W*0.1
    return X,Y

def train(trainX,trainY):
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    
    W = tf.Variable(np.random.randn(),name='weight')
    B = tf.Variable(np.random.randn(),name='bias')

    Y_pred = W*X + B 
    #也可以Y_pred = tf.add(tf.multiply(W,X) , b)
    
    loss = tf.reduce_sum(tf.square(Y - Y_pred))
    #也可以loss = tf.reduce_sum(tf.squared_difference(Y , Y_pred))

    #正则化
    loss = loss + 1e-6*W*W

    trainOp = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    epochs = 300 
    fig, ax = plt.subplots(1, 1)
    ax.scatter(trainX,trainY)
    fig.show()
    plt.draw()
    with tf.Session() as sess:
        epoch = 0
        sess.run(tf.initialize_all_variables())
        for epoch in range(epochs):
            for (x,y) in zip(trainX,trainY):
                t,w,b = sess.run([trainOp,W,B],feed_dict={X:x,Y:y})
            print("step:{},w:{},b:{}".format(epoch,w,b))

    print("final:W:{},b:{}".format(w,b))
    ax.plot(trainX,trainX*w + b)
    plt.show()

X,Y = createData(2,1)
train(X,Y)
        
