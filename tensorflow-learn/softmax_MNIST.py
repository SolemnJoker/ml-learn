import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data 
import matplotlib.pyplot as plt
import numpy as np
import sys

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
print("data num",mnist.train.num_examples,
      mnist.test.num_examples,
      mnist.validation.num_examples)
print("image shape:",mnist.train.images.shape)
print("label shape:",mnist.train.labels.shape)

#plt.imshow(np.reshape(mnist.train.images[100, :], (28, 28)), cmap='gray')
pix_num = mnist.train.images.shape[1]
label_num = mnist.train.labels.shape[1]

X = tf.placeholder(tf.float32,shape=[None,pix_num])
Y = tf.placeholder(tf.float32,shape=[None,label_num])

W = tf.Variable(tf.zeros([pix_num,label_num]))
b = tf.Variable(tf.zeros([label_num]))

Y_prev = tf.nn.softmax(tf.matmul(X,W)+b)
loss = -tf.reduce_sum(Y*tf.log(Y_prev))

train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#correct_prediction得到的是一个bool数组[True,False,True....]
correct_prediction = tf.equal(tf.argmax(Y_prev,1),tf.arg_max(Y,1))
#tf.cast将correct_prediction转换为浮点型数组
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

W_ = np.zeros([pix_num,label_num])
b_ = np.zeros([label_num])
batch_size = 50
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        for i in range(int(mnist.train.num_examples/batch_size)):
            batch = mnist.train.next_batch(batch_size)
            sess.run([train_op,W,b],feed_dict={X:batch[0],Y:batch[1]})
        
        print(sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))
        sys.stdout.flush()
    
    print(sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))
    batch = mnist.train.next_batch(1)
    W_,b_=sess.run([W,b],feed_dict={X:batch[0],Y:batch[1]})
    print(W_,W_.shape)
    sys.stdout.flush()
    #将每个w画出来，此处跟softmax无关
    plt.figure(1)
    for i in range(10):
        p = plt.subplot(3,4,i+1)
        p.imshow(np.reshape(W_[:,i], (28, 28)), cmap='gray')
    plt.show()
    
    
