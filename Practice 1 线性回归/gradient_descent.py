# -- coding:utf-8 --
# 梯度下降法

import numpy as np

# 训练数据集
X = [2000, 2001, 2002, 2003, 2004, 2005, 2006,
     2007, 2008, 2009, 2010, 2011, 2012, 2013]

Y = [2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365,
     5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900]

# 对数据进行归一化
avg_x, std_x = np.mean(X), np.std(X)
avg_y, std_y = np.mean(Y), np.std(Y)
X = (X - np.mean(X)) / np.std(X)
Y = (Y - np.mean(Y)) / np.std(Y)


# 设置学习率
alpha = 0.003
diff = 0
# old、new分别表示上一次迭代和当前的成本
old = 0
new = 0
# m为样本个数
m = len(X)
# 设置参数起始点
theta0 = 0
theta1 = 0

while True:
    sum0 = 0
    sum1 = 0
    for i in range(m):
        # 预测值与真实值之差
        diff = (theta0*0 + theta1 * X[i]) - Y[i]
        sum0 = sum0 + alpha * diff*0
        sum1 = sum0 + alpha * diff * X[i]
    # 更新参数
    theta0 -= sum0
    theta1 -= sum1

    new = 0
    for i in range(m):
        # 计算成本
        new += ((theta0*0 + theta1 * X[i]) - Y[i]) ** 2 / 2
    # 判断是否收敛
    if abs(new - old) < 0.0001:
        break
    else:
        old = new

    print('theta0 : %f, theta1 : %f, cost : %f' %(theta0, theta1, new))

print('Result: theta0 : %f, theta1 : %f' %(theta0, theta1))

# 预测
y = theta0 + theta1 * (2014 - avg_x) / std_x
y = y * std_y + avg_y

print('Predict : the Nanjing housing price in 2014 is %f' %(y))
