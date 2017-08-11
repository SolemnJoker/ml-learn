import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt
import sys

#初始化权重w,这里也是卷积核(filter)
def weight_variable(shape):
    #从截断的正态分布中输出随机值
    #shape:[filter_height, filter_width, in_channels, out_channels]
    #[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
    inital = tf.truncated_normal(shape)
    return tf.Variable(inital)

#初始化权重b
def bias_variable(shape):
    inital = tf.constant(0.1,shape=shape)
    return tf.Variable(inital)

#构建一个卷积层
def conv2d(x,w):
    #strides是卷积的步进,
    #[batch_size_stride,height_stride,width_stride,channels_stride]

    #padding:
    #SAME:卷积输出与输入的尺寸相同。这里在计算如何跨越图像时，并不考虑滤
    #波器的尺寸。选用该设置时，缺失的像素将用0填充，卷积核扫过的像素数将
    #超过图像的实际像素数。

    #VALID:在计算卷积核如何在图像上跨越时，需要考虑滤波器的尺寸。这会使
    # 卷积核尽量不越过图像的边界。 在某些情形下， 可能边界也会被填充。
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")

#池化层,简单的讲就是下采样，max_pool就是取区域内最大值
def max_pool(x):
    #ksize是池化窗口大小,[]列表里的参数和strides意义一致
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding = "SAME")


#读取数据
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
pix_num = mnist.train.images.shape[1]
label_num = mnist.train.labels.shape[1]

# 构建网络
X = tf.placeholder(tf.float32,shape=[None,pix_num])
Y = tf.placeholder(tf.float32,shape=[None,label_num])

#将MNIST的[784]数据转换为[28,28],-1表示None,我们不指定输入batch大小
x_image = tf.reshape(X,[-1,28,28,1]) 
w_conv1 = weight_variable([5,5,1,32])                    #生成32个特征
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1) #第一个卷积层，relu = max(0,x)
h_pool1 = max_pool(h_conv1)                           #第一个池化层

w_conv2 = weight_variable([5,5,32,64])                   
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)  #第二个卷积层
h_pool2 = max_pool(h_conv2)                           #第二个池化层

w_fc1 = weight_variable([7*7*64,1024])#生成1024个特征
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)  #第一个全连接层

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)              #dropout层

w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
Y_prev = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2) + b_fc2)

#训练
loss = -tf.reduce_sum(Y*tf.log(Y_prev))

train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#correct_prediction得到的是一个bool数组[True,False,True....]
correct_prediction = tf.equal(tf.argmax(Y_prev,1),tf.arg_max(Y,1))
#tf.cast将correct_prediction转换为浮点型数组
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

W_ = np.zeros([pix_num,label_num])
b_ = np.zeros([label_num])
batch_size = 50
print("start")
sys.stdout.flush()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        for i in range(int(mnist.train.num_examples/batch_size)):
            sys.stdout.flush()
            batch = mnist.train.next_batch(batch_size)
            sess.run([train_op],feed_dict={X:batch[0],Y:batch[1],keep_prob:0.5})
        
        batch = mnist.test.next_batch(1000)
        print(sess.run(accuracy,feed_dict={X:batch[0],Y:batch[1],keep_prob:1.0}))
        sys.stdout.flush()
    
    batch = mnist.test.next_batch(1000)
    print(sess.run(accuracy,feed_dict={X:batch[0],Y:batch[1],keep_prob:1.0}))






