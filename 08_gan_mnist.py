import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy

def get_inputs(real_img_size,noise_size):
    real_img = tf.placeholder(tf.float32,[None,real_img_size])
    noise_img = tf.placeholder(tf.float32,[None,noise_size])
    return real_img,noise_img

def get_discriminator(img,num_h,reuse=False,alpha=0.01):
    with tf.variable_scope("discriminator",reuse=reuse):
        h = tf.layers.dense(img,num_h)
        h = tf.maximum(alpha*h,h)
        
        logits = tf.layers.dense(h,1)
        outputs = tf.sigmoid(logits)
        return logits,outputs