import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy

def get_inputs(real_img_size,noise_size):
    real_img = tf.placeholder(tf.float32,[None,real_img_size])
    noise_img = tf.placeholder(tf.float32,[None,noise_size])
    return real_img,noise_img

def get_discriminator(img,num_h,reuse=False,alpha=0.01):
    '''
    img:输入图像
    n_units:判别器隐层数
    reuse:?
    alpha:leaky ReLU系数
    '''
    with tf.variable_scope("discriminator",reuse=reuse):
        h = tf.layers.dense(img,num_h)
        #Leaky ReLU系数
        h = tf.maximum(alpha*h,h)

        logits = tf.layers.dense(h,1)
        outputs = tf.sigmoid(logits)
        return logits,outputs

def get_generator(noise_data,n_units,out_dim,reuse=False,alpha=0.01):
    '''
    noise_data:输入的噪声数据，可以和out_dim不一样大
    n_units:生成器隐层数
    out_dim：生成图片的大小
    reuse:?
    alpha:leaky ReLU系数
    '''
    with tf.variable_scope("generator",reuse=reuse):
        hidden1 = tf.layers.dense(noise_data,n_units)
        hidden1 = tf.maximum(alpha*hidden1,hidden1)

        hidden1 = tf.layers.dropout(hidden1,rate=0.2)

        logits = tf.layers.dense(hidden1,out_dim)
        outputs = tf.tanh(logits)
        return logits,outputs

#训练用的参数    
mnist = input_data.read_data_sets("/MNIST_data",one_hot=True)
img_size = mnist.train.images[0].shape[0]
#噪声数据大小
noise_size = 100
g_units = 128
d_units = 128
#leaky ReLU参数
alpha=0.01
learning_rate=0.001
#label smoothing??
smooth=0.1

#构建网络
tf.reset_default_graph()
real_img,noise_data = get_inputs(img_size,noise_size)

#generator
g_logits,g_outputs = get_generator(noise_data,g_units,img_size)

#discriminator
d_logits_real,d_outputs_real = get_discriminator(real_img,d_units)
d_logits_fake,d_outputs_fake = get_discriminator(g_outputs,d_units,reuse=True)


