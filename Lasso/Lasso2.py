import numpy as np
from sklearn import linear_model

np.random.seed(42)
N = 6 #サンプル数
x = np.linspace(-3, 3, N)
X = np.zeros((N, 5)) #xの次数が5までの基底関数
X[:, 0] = x
for i in range(4):
    X[:, i+1] = X[:, i] * X[:, 0]
y = np.sin(X[:, 0]).reshape(-1,1) + np.random.randn(N, 1)/3 + 1.5


import matplotlib.pyplot as plt

xx = np.linspace(-3,3,20)

reg_lasso = linear_model.Lasso(alpha=0.1)
reg_lasso.fit(X, y)

XX = np.zeros((20,5)) 
XX[:,0] = xx
for i in range(4):
    XX[:, i+1] = XX[:,i]*XX[:,0]
 
plt.scatter(X[:,0], y, label='raw data')
plt.plot(xx, np.sin(xx)+1.5, label='sin curve')
plt.plot(xx, reg_lasso.predict(XX), label='lasso(α=0.1)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()
