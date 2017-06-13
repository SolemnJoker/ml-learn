import numpy as np
import matplotlib.pyplot as plt

N = 100
K = 3
D = 2
X = np.zeros((N * K, D))
Y = np.zeros(N * K, dtype='uint8')
for k in range(K):
    idx = range(N * k, N * (k + 1))
    r = np.linspace(0.0, 1.0, N)
    t = np.linspace(4 * k, (k + 1) * 4, N) + np.random.randn(N) * 0.2
    X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
    Y[idx] = k

plt.scatter(X[:, 0], X[:, 1])
