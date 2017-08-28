import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import sys
import time
import pickle
import matplotlib.pyplot as plt

def get_inputs(real_img_size,noise_size):
    real_img = tf.placeholder(tf.float32,[None,real_img_size],name="real_image")
    noise_img = tf.placeholder(tf.float32,[None,noise_size],name="noise_image")
    return real_img,noise_img

def get_discriminator(img,num_h,reuse=False,alpha=0.01):
    '''
    img:输入图像
    n_units:判别器隐层数
    reuse:?
    alpha:leaky ReLU系数
    '''
    with tf.variable_scope("discriminator",reuse=reuse):
        h = tf.layers.dense(img,num_h,name="hidden_layer")
        #Leaky ReLU系数
        h = tf.maximum(alpha*h,h)

        logits = tf.layers.dense(h,1,name="logits")
        outputs = tf.sigmoid(logits,name="output")
        return outputs

def get_generator(noise_data,n_units,out_dim,reuse=False,alpha=0.01):
    '''
    noise_data:输入的噪声数据，可以和out_dim不一样大
    n_units:生成器隐层数
    out_dim：生成图片的大小
    reuse:?
    alpha:leaky ReLU系数
    '''
    with tf.variable_scope("generator",reuse=reuse):
        hidden1 = tf.layers.dense(noise_data,n_units,name="hidden_layer")
        hidden1 = tf.maximum(alpha*hidden1,hidden1)

        hidden1 = tf.layers.dropout(hidden1,rate=0.2)

        logits = tf.layers.dense(hidden1,out_dim,name="logits")
        outputs = tf.tanh(logits)
        return outputs

#训练用的参数    
mnist = input_data.read_data_sets("/MNIST_data",one_hot=True)
img_size = mnist.train.images[0].shape[0]
#噪声数据大小
noise_size = 100
g_units = 128
d_units = 128
#leaky ReLU参数
alpha=0.01
learning_rate=0.0015
#label smoothing??
smooth=0.1

#构建网络
tf.reset_default_graph()
real_img,noise_data = get_inputs(img_size,noise_size)

#generator
g_outputs = get_generator(noise_data,g_units,img_size)

#discriminator
d_outputs_real = get_discriminator(real_img,d_units)
d_outputs_fake = get_discriminator(g_outputs,d_units,
reuse=True)

#Loss 1-smooth??
#真实图片loss
# tf.ones_like生成和tensor shape一样的全为1的tensor，zero_like同理
# 'tensor' is [[1, 2, 3], [4, 5, 6]] 
#  tf.ones_like(tensor) ==> [[1, 1, 1], [1, 1, 1]]
d_loss_real = tf.reduce_mean(tf.log(d_outputs_real)) *(1-smooth)

#生成图片loss
d_loss_fake =tf.reduce_mean(tf.log(1-d_outputs_fake))*(1-smooth)

#总的discriminator loss = -1/m * Σ(log(D(x)) +log(1-D(G())))
d_loss = -(d_loss_fake +  d_loss_real)

#generator loss = 1/m * Σ(log(1-D(G())))
#这里使用-log(D(G()))替代原公式是为了在训练初期加大梯度信息
g_loss = -tf.reduce_mean(tf.log(d_outputs_fake)) *(1-smooth)


#Optimizer
#tf.trainable_variables返回的是需要训练的变量列表
train_vars = tf.trainable_variables()
g_var = [var for var in train_vars if var.name.startswith("generator")]
d_var = [var for var in train_vars if var.name.startswith("discriminator")]

#optimizer
d_train_op = tf.train.AdamOptimizer(learning_rate).minimize(d_loss,var_list=d_var)
g_train_op = tf.train.AdamOptimizer(learning_rate).minimize(g_loss,var_list=g_var)

#训练
batch_size = 50
epochs = 300
n_sample = 25
samples = []
losses = []
saver = tf.train.Saver(var_list=g_var)

with tf.Session() as sess:
    with tf.device("/cpu:0"):
        #tensorboard:添加变量到summary中
        tf.summary.scalar("d_loss_real",d_loss_real)
        tf.summary.scalar("d_loss_fake",d_loss_fake)
        tf.summary.scalar("d_loss",d_loss)
        tf.summary.scalar("g_loss",g_loss)
        #tensorboard:merge summary
        merge = tf.summary.merge_all()
        #tensorboard:创建writer
        writer = tf.summary.FileWriter("./gan_mnist", sess.graph)
 
        start_time = time.clock()
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            epoch_start = time.clock()
            for batch_i in range(int(mnist.train.num_examples/batch_size)):
                batch = mnist.train.next_batch(batch_size)
                batch_images = batch[0].reshape((batch_size,784))
                #tanh输出（-1，1）转换到（0，1）
                batch_images = batch_images*2 - 1
                batch_noise = np.random.uniform(-1,1,size=(batch_size,noise_size))
                sess.run(d_train_op,feed_dict={real_img:batch_images,noise_data:batch_noise})
                sess.run(g_train_op,feed_dict={noise_data:batch_noise})


            train_loss_d,train_loss_d_real,train_loss_d_fake,train_loss_g = sess.run(
                [d_loss,d_loss_real,d_loss_fake,g_loss],feed_dict= {
                    real_img:batch_images, noise_data:batch_noise})

            result = sess.run(merge,feed_dict={real_img:batch_images,
                noise_data:batch_noise})
            writer.add_summary(result,e)
 

            print("Epoch {}/{}...".format(e+1, epochs),
              "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(train_loss_d, train_loss_d_real, train_loss_d_fake),
              "Generator Loss: {:.4f}".format(train_loss_g))
            sys.stdout.flush()

            # 抽取样本后期进行观察
            sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))
            gen_samples = sess.run(get_generator(noise_data, g_units, img_size, reuse=True),
                                   feed_dict={noise_data: sample_noise})
            samples.append(gen_samples)
            epoch_end = time.clock()
            print("epoch time:{}".format(epoch_end - epoch_start))
            sys.stdout.flush()

            # 存储checkpoints
            saver.save(sess, './checkpoints/generator.ckpt')
        end_time = time.clock()
        sys.stdout.flush()
        print("total time:{}".format(end_time - start_time))

# 将sample的生成数据记录下来
with open('train_samples.pkl', 'wb') as f:
    pickle.dump(samples, f)
 
       
with open('train_samples.pkl', 'rb') as f:
    samples = pickle.load(f)

def view_samples(epoch, samples):
    """
    epoch代表第几次迭代的图像
    samples为我们的采样结果
    """
    fig, axes = plt.subplots(figsize=(7,7), nrows=5, ncols=5, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]): # 这里samples[epoch][1]代表生成的图像结果，而[0]代表对应的logits
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')
    
    return fig, axes

_ = view_samples(-1, samples) # 显示最后一轮的outputs

# 指定要查看的轮次
epoch_idx = [0, 5, 10, 20, 40, 60, 80, 100, 150, 250] # 一共300轮，不要越界
show_imgs = []
for i in epoch_idx:
    show_imgs.append(samples[i])

# 指定图片形状
rows, cols = 10, 25
fig, axes = plt.subplots(figsize=(30,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

idx = range(0, epochs, int(epochs/rows))

for sample, ax_row in zip(show_imgs, axes):
    for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
plt.show()





