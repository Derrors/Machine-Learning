# -- coding:utf-8 --
# 解析式法

import numpy as np

# 导入数据
X = np.array([[1, 2000], [1, 2001], [1, 2002], [1, 2003], [1, 2004], [1, 2005], [1, 2006],
              [1, 2007], [1, 2008], [1, 2009], [1, 2010], [1, 2011], [1, 2012], [1, 2013]])

Y = np.array([2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365,
              5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900]).reshape(-1, 1)

# 将数据集转换为矩阵形式
X, Y = np.mat(X), np.mat(Y)

# 计算导数为零时的参数
threa = (X.T.dot(X)).I.dot(X.T).dot(Y)

print('Result: threa0 : %f, threa1 : %f' %(threa[0], threa[1]))

# 预测
y = threa[0] + threa[1] * 2014

print('Predict : the Nanjing housing price in 2014 is %f' %(y))
