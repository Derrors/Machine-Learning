# -- coding:utf-8 --
# 使用sklearn包进行预测

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 导入数据并转换为列向量
X = np.array([2000, 2001, 2002, 2003, 2004, 2005, 2006,
              2007, 2008, 2009, 2010, 2011, 2012, 2013]).reshape(-1, 1)

Y = np.array([2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365,
              5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900]).reshape(-1, 1)

# 划分训练集与测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=3)

# 设置使用的模型
model = LinearRegression()
# 使用模型进行训练
model.fit(X_train, Y_train)
# 计算训练得分
train_score = model.score(X_train, Y_train)
# 计算测试分数
cv_score = model.score(X_test, Y_test)

# 输出结果并进行预测
print('train_score: %f; cv_score: %f' %(train_score, cv_score))
print(model.predict(np.array([2014]).reshape(-1, 1)))
