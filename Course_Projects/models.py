# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from sklearn.model_selection import KFold


class Linear_Regression():
    def __init__(self, input_x, input_y):
        self.x = input_x
        self.y = input_y
        self.weight = np.random.rand(self.x.shape[1], 1)

    def loss_fn(self, pred):
        loss = np.sum(np.square(pred - self.y)) / 2
        grad = np.matmul(self.x.T, (pred - self.y))
        return loss, grad

    def close_form_solution(self):
        self.weight = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.x.T, self.x)), self.x.T), self.y)

    def gradient_descent(self, learning_rate, epochs):
        for e in range(epochs):
            pred = np.matmul(self.x, self.weight)
            loss, grad = self.loss_fn(pred)
            self.weight -= learning_rate * grad

    def predict_new(self, x=[1, 1.86]):
        pred = np.matmul(x, self.weight)
        return pred


class Logistic_Regression():
    def __init__(self, input_x, input_y):
        self.x = input_x
        self.y = input_y
        self.weight = np.random.rand(self.x.shape[1], 1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss_fn(self, x, y, pred):
        loss = - np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))
        grad = np.matmul(x.T, (pred - y))
        return loss, grad

    def calculate_accuracy(self):
        pred = self.sigmoid(np.matmul(self.x, self.weight))
        pred = np.asarray((pred >= 0.5), dtype=int)
        accuracy = np.sum(pred == self.y) / self.x.shape[0]
        return accuracy

    def gradient_descent(self, learning_rate, epochs):
        for e in range(epochs):
            pred = self.sigmoid(np.matmul(self.x, self.weight))
            loss, grad = self.loss_fn(self.x, self.y, pred)
            self.weight -= learning_rate * grad
            print(loss)

    def stochastic_gradient_descent(self, learning_rate, epochs):
        orders = list(range(self.x.shape[0]))
        random.seed(2020)
        random.shuffle(orders)
        for e in range(epochs):
            for index in orders:
                pred = self.sigmoid(np.matmul(self.x[index], self.weight))
                _, grad = self.loss_fn(self.x[index].reshape(1, -1), self.y[index].reshape(-1, 1), pred.reshape(-1, 1))
                self.weight -= learning_rate * grad

    def newton_method(self, epochs):
        for e in range(epochs):
            pred = self.sigmoid(np.matmul(self.x, self.weight)).reshape(-1, 1)
            loss, grad = self.loss_fn(self.x, self.y, pred)
            H = np.matmul(self.x.T, pred * (1 - pred) * self.x)
            self.weight -= np.matmul(np.linalg.inv(H), grad)
            print(loss)


class Softmax_Regression():
    def __init__(self, input_x, input_y):
        self.x = input_x
        self.y = input_y
        self.weight = np.random.rand(self.x.shape[1], self.y.shape[1])

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)

    def loss_fn(self, x, y, pred):
        loss = - np.sum(y * np.log(pred))
        grad = np.matmul(x.T, pred - y)
        return loss, grad

    def calculate_accuracy(self):
        pred = self.softmax(np.matmul(self.x, self.weight))
        pred = np.argmax(pred, axis=1)
        true = np.argmax(self.y, axis=1)
        accuracy = np.sum(pred == true) / self.x.shape[0]
        return accuracy

    def gradient_descent(self, learning_rate, epochs):
        for e in range(epochs):
            pred = self.softmax(np.matmul(self.x, self.weight))
            loss, grad = self.loss_fn(self.x, self.y, pred)
            self.weight -= learning_rate * grad
            print(loss)

    def stochastic_gradient_descent(self, learning_rate, epochs):
        orders = list(range(self.x.shape[0]))
        random.seed(2020)
        random.shuffle(orders)
        for e in range(epochs):
            for index in orders:
                pred = np.matmul(self.x[index], self.weight)
                pred = (np.exp(pred) / np.sum(np.exp(pred)))
                _, grad = self.loss_fn(self.x[index].reshape(1, -1), self.y[index].reshape(1, -1), pred.reshape(1, -1))
                self.weight -= learning_rate * grad


class Perceptron():
    def __init__(self, input_x, input_y):
        self.x = input_x
        self.y = input_y
        self.weight = np.random.rand(self.x.shape[1], 1)

    def hypothesis(self, pred):
        pred = np.asarray(pred >= 0, dtype=int)
        return pred
    
    def loss_fn(self, x, y, pred):
        loss = np.sum(np.multiply(np.matmul(pred - y, self.weight.T), x))
        grad = np.matmul(x.T, pred- y)
        return loss, grad

    def calculate_accuracy(self):
        pred = self.hypothesis(np.matmul(self.x, self.weight))
        accuracy = np.sum(pred == self.y) / self.x.shape[0]
        return accuracy

    def stochastic_gradient_descent(self, learning_rate, epochs):
        orders = list(range(self.x.shape[0]))
        random.seed(2020)
        random.shuffle(orders)
        for e in range(epochs):
            for index in orders:
                pred = self.hypothesis(np.matmul(self.x[index], self.weight))
                _, grad = self.loss_fn(self.x[index].reshape(1, -1), self.y[index].reshape(1, -1), pred.reshape(1, -1))
                self.weight -= learning_rate * grad


class Multi_Class_Perceptron():
    def __init__(self, input_x, input_y):
        self.x = input_x
        self.y = input_y
        self.weight = np.random.rand(self.x.shape[1], self.y.shape[1])

    def loss_fn(self, x, y, pred):
        loss = np.sum(np.multiply(np.matmul(pred - y, self.weight.T), x))
        grad = np.matmul(x.T, pred- y)
        return loss, grad

    def calculate_accuracy(self):
        pred = np.argmax(np.matmul(self.x, self.weight), axis=1)
        true = np.argmax(self.y, axis=1)
        accuracy = np.sum(pred == true) / self.x.shape[0]
        return accuracy

    def stochastic_gradient_descent(self, learning_rate, epochs):
        orders = list(range(self.x.shape[0]))
        random.seed(2020)
        random.shuffle(orders)
        for e in range(epochs):
            for index in orders:
                pred = np.matmul(self.x[index], self.weight)
                _, grad = self.loss_fn(self.x[index].reshape(1, -1), self.y[index].reshape(1, -1), pred.reshape(1, -1))
                self.weight -= learning_rate * grad


class Artifical_Neural_Network():
    def __init__(self, input_x, input_y, hidden_num):
        self.x = input_x
        self.y = input_y
        self.w1 = np.random.rand(self.x.shape[1], hidden_num)
        self.b1 = np.random.rand(1, hidden_num)
        self.w2 = np.random.rand(hidden_num, self.y.shape[1])
        self.b2 = np.random.rand(1, self.y.shape[1])

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def calculate_accuracy(self, x, y):
        pred = self.sigmoid(np.matmul(self.sigmoid(np.matmul(x, self.w1) + self.b1), self.w2) + self.b2)
        pred = np.argmax(pred, axis=1)
        true = np.argmax(y, axis=1)
        accuracy = np.sum(pred == true) / x.shape[0]
        return accuracy

    def loss_fn(self, x, y, output_hidden, output):
        loss = 0.5 * np.sum(np.square(output - y))
        error_output = (output - y) * output * (1 - output)
        error_hidden = np.matmul(error_output, self.w2.T) * output_hidden * (1 - output_hidden)
        grad_w2 = np.matmul(output_hidden.T, error_output)
        grad_b2 = np.sum(error_output, axis=0)
        grad_w1 = np.matmul(x.T, error_hidden)
        grad_b1 = np.sum(error_hidden, axis=0)
        return loss, grad_w2, grad_b2, grad_w1, grad_b1
    
    def gradient_descent(self, learning_rate, epochs, n_folds):
        kfold = KFold(n_splits=n_folds, random_state=1)
        for e in range(epochs):
            for k, (train, test) in enumerate(kfold.split(self.x)):
                output_hidden = self.sigmoid(np.matmul(self.x[train], self.w1) + self.b1)
                output = self.sigmoid(np.matmul(output_hidden, self.w2) + self.b2)
                loss, grad_w2, grad_b2, grad_w1, grad_b1 = self.loss_fn(self.x[train], self.y[train], output_hidden, output)
                self.w2 -= learning_rate * grad_w2
                self.b2 -= learning_rate * grad_b2
                self.w1 -= learning_rate * grad_w1
                self.b1 -= learning_rate * grad_b1
            accuracy = self.calculate_accuracy(self.x, self.y)
            print(accuracy)


class ANN(nn.Module):
    def __init__(self, input_size, output_size, n_hiddens):
        super().__init__()

        linear_layers = []
        n_hiddens.insert(0, input_size)
        n_hiddens.append(output_size)
        for i in range(len(n_hiddens) - 1):
            linear_layers.append(nn.Linear(n_hiddens[i], n_hiddens[i+1], bias=True))
            linear_layers.append(nn.ReLU())

        self.ann = nn.Sequential(*linear_layers)
    
    def forward(self, x):
        return self.ann(x)



