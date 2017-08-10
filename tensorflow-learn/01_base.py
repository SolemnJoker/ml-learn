#coding:utf-8
import tensorflow as tf

#constant常量
def tensor():
    a = tf.constant([1.0,2.0],name='a')#name可以在tensorboard中显示
    b = tf.constant([2.0,3.0],name='b')
    res = a + b # res is a tensor
    print(res.get_shape())

    #计算 session,
    # 注意图过session close,res.eval会无效
    sess = tf.Session()
    res_eval = sess.run(res)
    print("on with",res.eval(session=sess),res_eval)
    sess.close()


    #with sess
    with tf.Session() as sess:
        sess.run(res)
        print("with:",res.eval())
        tensorboard(sess)

def variable():
    state = tf.Variable(0,name='var1')
    one = tf.constant(1)
    new_val = tf.add(state,one)
    update = tf.assign(state,new_val)

    #初始化变量
    init_op = tf.initialize_all_variables()

    print(type(state))
    print(type(update))
    print(type(init_op))

    with tf.Session() as sess:
        sess.run(init_op)
        print(sess.run(state))
        for i in range(3):
            print("update:",sess.run(update))
            print("state:",sess.run(state))
            tensorboard(sess)
        
#一个会话多个计算和取回，
def fetch():
    input1 = tf.constant(1.)
    input2 = tf.constant(3.)
    input3 = tf.constant(4.)
    add = tf.add(input2,input3)
    mul = tf.multiply(input1,add)

    with tf.Session() as sess:
        res = sess.run([mul,add])
        print(res)
        print(type(res))  #返回list
        tensorboard(sess)
    
def feed():
    #使用placeholder占位符
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    out = tf.multiply(input1,input2)
    with tf.Session() as sess:
        print(sess.run([out],feed_dict={input1:[2.],input2:[4.]}))
        tensorboard(sess)
    
    
    #不使用占位符也可以用feed_dict替换
    a = tf.add(2,3)
    b = tf.multiply(a,5)
    replace_dict = {a:8}
    with tf.Session() as sess:
        print(sess.run(b,feed_dict=replace_dict))
        tensorboard(sess)
 
def nameScope():
    with tf.name_scope("scopeA"):
        a = tf.add(1,2,name='a_add')
        b = tf.reduce_prod([1,2,3,4,5,6],name='a_mul')

    with tf.name_scope("scopeB"):
        c = tf.reduce_sum([1,2,3,4],name='a_add')
        d = tf.multiply(1,2,name='a_mul')

    e = tf.add(b,d)
    writer = tf.summary.FileWriter("./my_graph", graph = tf.get_default_graph())
    writer.close()
 


#控制台:tensorboard --logdir="./my_graph" #启动tensorboard服务，会使用6006端口
#浏览器:http://localhost:6006 #最好用chrome，点击顶部graph看到数据流图
def tensorboard(sess):
    logdir = "./my_graph"
    writer = tf.summary.FileWriter(logdir, sess.graph)
    writer.close()
    
    
tensor()
fetch()
feed()
nameScope()




    
    
