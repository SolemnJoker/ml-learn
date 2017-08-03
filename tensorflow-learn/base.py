#coding:utf-8
import tensorflow as tf

def tensor():
    a = tf.constant([1.0,2.0],name='a')
    b = tf.constant([2.0,3.0],name='b')
    res = a + b # res is a tensor
    print res.get_shape()

    #计算 session,
    # 注意图过session close,res.eval会无效
    sess = tf.Session()
    res_eval = sess.run(res)
    print "on with",res.eval(session=sess),res_eval
    sess.close()


    #with sess
    with tf.Session() as sess:
        sess.run(res)
        print "with:",res.eval()

def variable():
    state = tf.Variable(0,name='var1')
    one = tf.constant(1)
    new_val = tf.add(state,one)
    update = tf.assign(state,new_val)

    #初始化变量
    init_op = tf.initialize_all_variables()

    print type(state)
    print type(update)
    print type(init_op)

    with tf.Session() as sess:
        sess.run(init_op)
        print sess.run(state)
        for i in range(3):
            print "update:",sess.run(update)
            print "state:",sess.run(state)
        
def fetch():
    input1 = tf.constant(1.)
    input2 = tf.constant(3.)
    input3 = tf.constant(4.)
    add = tf.add(input2,input3)
    mul = tf.multiply(input1,add)

    with tf.Session() as sess:
        res = sess.run([mul,add])
        print res
    
def feed():
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    out = tf.multiply(input1,input2)
    with tf.Session() as sess:
        print sess.run([out],feed_dict={input1:[2.],input2:[4.]})
    


    
    