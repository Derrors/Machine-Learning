# -*- coding: utf-8 -*-

import numpy as np
import torch.nn
import torch
import torch.optim as optim
from models import *


def read_data(file_path_x, file_path_y):
    x = np.loadtxt(file_path_x, dtype=float)
    y = np.loadtxt(file_path_y, dtype=int)
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


if __name__ == "__main__":
    x = np.loadtxt('./data/iris_x.dat', dtype=float, ndmin=2)
    y = np.loadtxt('./data/iris_y.dat', dtype=int, ndmin=2)
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



'''
Project 1:

    x = np.loadtxt('./data/year_x.dat', dtype=float, ndmin=2)
    y = np.loadtxt('./data/price_y.dat', dtype=float, ndmin=2)
    x = data_preprocess(x)
    model = Linear_Regression(x, y)
    model.gradient_descent(0.03, 100)
    model.predict_new()

Project 2:

    x = np.loadtxt('./data/exam_x.dat', dtype=float, ndmin=2)
    y = np.loadtxt('./data/exam_y.dat', dtype=int, ndmin=2)
    x = data_preprocess(x)
    model = Logistic_Regression(x, y)
    model.gradient_descent(0.05, 200)
    print(model.calculate_accuracy())

Project 3:
    x = np.loadtxt('./data/exam_x.dat', dtype=float, ndmin=2)
    y = np.loadtxt('./data/exam_y.dat', dtype=int, ndmin=2)
    x = data_preprocess(x)
    y = convert_to_one_hot(y)
    model = Softmax_Regression(x, y)
    model.gradient_descent(0.001, 200)
    print(model.calculate_accuracy())

Project 4:
    x = np.loadtxt('./data/exam_x.dat', dtype=float, ndmin=2)
    y = np.loadtxt('./data/exam_y.dat', dtype=int, ndmin=2)
    x = data_preprocess(x)
    model = Perceptron(x, y)
    model.stochastic_gradient_descent(0.001, 100)
    print(model.calculate_accuracy())

    x = np.loadtxt('./data/iris_x.dat', dtype=float, ndmin=2)
    y = np.loadtxt('./data/iris_y.dat', dtype=int, ndmin=2)
    x = data_preprocess(x)
    y = convert_to_one_hot(y)
    model = Multi_Class_Perceptron(x, y)
    model.stochastic_gradient_descent(0.001, 100)
    print(model.calculate_accuracy())

Project 5:
    x = np.loadtxt('./data/exam_x.dat', dtype=float, ndmin=2)
    y = np.loadtxt('./data/exam_y.dat', dtype=int, ndmin=2)
    x = data_preprocess(x)
    y = convert_to_one_hot(y)
    model = Artifical_Neural_Network(x, y, 10)
    model.gradient_descent(0.003, 200, 5)

Project 6:
    x = np.loadtxt('./data/iris_x.dat', dtype=float, ndmin=2)
    y = np.loadtxt('./data/iris_y.dat', dtype=int, ndmin=2)
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

'''
