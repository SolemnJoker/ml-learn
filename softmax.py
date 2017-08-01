#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt


N = 100
K = 3
D = 2
X = np.zeros((N * K, D))
Y = np.zeros(N * K, dtype='uint8')
reg = 1e-3
step_size = 1

def createData():
    for k in range(K):
        idx = range(N * k, N * (k + 1))
        r = np.linspace(0.0, 1.0, N)
        t = np.linspace(4 * k, (k + 1) * 4, N) + np.random.randn(N) * 0.2
        X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[idx] = k


def softmax(X):
    for i in range(200):
        num_examples = X.shape[0]
        W = 0.01 * np.random.randn(D, K)
        b = np.zeros((1, K))
        #计算类别得分,结果为[N x K]矩阵
        scores = np.dot(X, W) + b
    
        #计算类别概率,结果为[N x K]矩阵
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

         # 计算损失loss(包括互熵损失和正则化部分)结果为N x 1数组
        corect_logprobs = -np.log(probs[range(num_examples),Y])
        data_loss = np.sum(corect_logprobs)/num_examples
        reg_loss = 0.5*reg*np.sum(W*W)
        loss = data_loss + reg_loss
        print loss

        # 计算得分上的梯度
        dscores = probs
        dscores[range(num_examples),Y] -=1 #梯度 = pk- 1(yi == k)
        dscores /= num_examples

        # 计算和回传梯度
        dW = np.dot(X.T,dscores)
        db = np.sum(dscores,axis=0,keepdims =True)
        dW += reg*W

        # 更新参数
        W += -step_size*dW  
        b += -step_size*db  

    return W,b

def show(W,b):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.show()

createData()
W,b = softmax(X)
show(W,b)
