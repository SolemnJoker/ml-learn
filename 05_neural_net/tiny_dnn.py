#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import sys
N = 100
K = 3
D = 2
 
def createData():
    X = np.zeros((N * K, D))
    Y = np.zeros(N * K, dtype='uint8')
    for k in range(K):
        idx = range(N * k, N * (k + 1))
        r = np.linspace(0.0, 1.0, N)
        t = np.linspace(4 * k, (k + 1) * 4, N) + np.random.randn(N) * 0.2
        X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[idx] = k
    return X,Y


def train(X,Y):
    h = 100 
    W = 0.01*np.random.randn(D,h)
    b = np.zeros((1,h))
    W2 = 0.01*np.random.randn(h,K)
    b2 = np.zeros((1,K))
 
    step = 0.1
    reg = 0.0002
    num = X.shape[0]
    for i in range(30000):
        h1 = np.maximum(0,np.dot(X,W) + b)
        scores = np.dot(h1,W2) + b2

        exp_scores = np.exp(scores)
        probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
        corect_logprobs = -np.log(probs[range(num),Y])
        data_loss = np.sum(corect_logprobs)/num
        reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
        loss = data_loss + reg_loss
        if i % 50 ==0:
            print("{} , loss:{}".format(i,loss))
            sys.stdout.flush()
        # 计算梯度
        dscores = probs
        dscores[range(num),Y] -= 1
        dscores /= num
        
        # 梯度回传
        dW2 = np.dot(h1.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
        
        dhidden = np.dot(dscores, W2.T)
        
        dhidden[h1 <= 0] = 0
        # 拿到最后W,b上的梯度
        dW = np.dot(X.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)
        
        # 加上正则化梯度部分
        dW2 += reg * W2
        dW += reg * W
        
        # 参数迭代与更新
        W += -step * dW
        b += -step * db
        W2 += -step * dW2
        b2 += -step * db2
    
    return W,b,W2,b2
    
    


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
