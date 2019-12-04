# -*- coding: utf-8 -*-

import re
import os
import numpy as np
import torch
import torch.nn
import torch.optim as optim
from models import *


def str_filter(string):
    chars = re.compile(r'[^\u4e00-\u9fa5]')
    return re.sub(chars, '', string)

def read_data(name):
    if name == 'price':
        x = np.loadtxt('./data/year_x.dat', dtype=float, ndmin=2)
        y = np.loadtxt('./data/price_y.dat', dtype=int, ndmin=2)
    elif name == 'exam':
        x = np.loadtxt('./data/exam_x.dat', dtype=float, ndmin=2)
        y = np.loadtxt('./data/exam_y.dat', dtype=int, ndmin=2)
    else:
        x = np.loadtxt('./data/iris_x.dat', dtype=float, ndmin=2)
        y = np.loadtxt('./data/iris_y.dat', dtype=int, ndmin=2)
    return x, y

def data_normalize(x):
    μ = np.mean(x, axis=0)
    σ = np.std(x, axis=0)
    return (x - μ) / σ

def data_preprocess(x):
    _x = np.ones((x.shape[0], x.shape[1] + 1))
    _x[:, 1:]= data_normalize(x)
    return _x

def convert_to_one_hot(y):
    _y = np.zeros((y.shape[0], np.max(y) + 1))
    for i in range(y.shape[0]):
        _y[i, y[i]] = 1
    return _y

def build_input(path):
    stop_words = []
    with open('./data/Tsinghua/stop_words_zh.txt', 'r', encoding='gb18030', errors='ignore') as fs:
        lines = fs.readlines()
        for line in lines:
            words = line.strip().split()
            stop_words.extend(words)
    stop_words = set(stop_words)

    datas = os.listdir(path)
    x = []
    for data in datas:
        file_path = os.path.join(path, data)
        with open(file_path, 'r', encoding='gb18030', errors='ignore') as fd:
            lines = fd.readlines()
            docs = []
            for line in lines:
                doc = []
                line.lstrip('<text>')
                line.lstrip('</text>')
                tokens = line.split()
                if len(tokens) > 0:
                    for token in tokens:
                        token = str_filter(token)
                        if len(token) > 0 and token not in stop_words:
                            doc.append(token)
                if len(doc) > 0:
                    docs.append(doc)
        x.append(docs)
    y = range(0, len(datas))
    
    return x, y

def Project_1():
    x, y = read_data('price')
    x = data_preprocess(x)
    model = Linear_Regression(x, y)
    model.gradient_descent(0.003, 100)
    model.predict_new()

def Project_2():
    x, y = read_data('exam')
    x = data_preprocess(x)
    model = Logistic_Regression(x, y)
    model.gradient_descent(0.003, 150)
    # model.stochastic_gradient_descent(0.1, 5)
    # model.newton_method(20)
    print(model.calculate_accuracy())

def Project_3():
    x, y = read_data('iris')
    x = data_preprocess(x)
    y = convert_to_one_hot(y)
    model = Softmax_Regression(x, y)
    #model.gradient_descent(0.001, 150)
    model.stochastic_gradient_descent(0.03, 3)
    print(model.calculate_accuracy())

def Project_4_1():
    x, y = read_data('exam')
    x = data_preprocess(x)
    model = Perceptron(x, y)
    model.stochastic_gradient_descent(0.03, 5)
    print(model.calculate_accuracy())

def Project_4_2():
    x, y = read_data('iris')
    x = data_preprocess(x)
    y = convert_to_one_hot(y)
    model = Multi_Class_Perceptron(x, y)
    model.stochastic_gradient_descent(0.03, 2)
    print(model.calculate_accuracy())   

def Project_5_1():
    x, y = read_data('iris')
    x = data_preprocess(x)
    y = convert_to_one_hot(y)
    model = Artifical_Neural_Network(x, y, 10)
    model.gradient_descent(0.03, 200, 5)

def Project_6():
    x, y = read_data('iris')
    x = torch.tensor(data_preprocess(x), dtype=torch.float)
    y = torch.tensor(convert_to_one_hot(y), dtype=torch.long)

    model = ANN(x.shape[1], y.shape[1], [10, 10]) 
    criterion = nn.MSELoss()
    optimzer = optim.Adam(model.parameters(), lr=0.03)

    model.train()
    for e in range(200):
        optimzer.zero_grad()

        output = model(x)
        loss = criterion(output, torch.tensor(y, dtype=torch.float))
        pred = np.argmax(output.detach().numpy(), axis=1)
        true = np.argmax(y.detach().numpy(), axis=1)
        accuracy = np.sum(pred == true) / x.shape[0]
        
        loss.backward()
        optimzer.step()
        print(loss, accuracy)
    
def Project_7():
    train_x, train_y = build_input('./data/Tsinghua/train/')
    test_x, test_y = build_input('./data/Tsinghua/test/')
    
    model = Naive_Bayes(train_x, train_y)
    model.multinomial(test_x, test_y)


if __name__ == "__main__":
    Project_7()
