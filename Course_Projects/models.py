# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

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
        self.plot()
        plt.ion()
        for e in range(epochs):
            pred = np.matmul(self.x, self.weight)
            loss, grad = self.loss_fn(pred)
            self.weight -= learning_rate * grad

            # 以下为画图代码
            plt.cla()
            self.epoch.append(e)
            self.loss.append(loss)
            self.ax1.plot(self.epoch, self.loss, 'r')
            
            y1 = self.predict_new(self.x[0])
            y2 = self.predict_new(self.x[-1])
            self.ax2.scatter(self.x[:, 1], self.y, c='b')
            self.ax2.plot([self.x[0, 1], self.x[-1, 1]], [y1, y2], 'r')
            plt.pause(0.01)
        plt.ioff()
        plt.show()

        
    def predict_new(self, x=[1, 1.86]):
        pred = np.matmul(x, self.weight)
        return pred

    def plot(self):
        self.fig = plt.figure(num=1, figsize=(15, 8),dpi=80)
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax1.set_title('The loss of Linear_Regression')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.loss, self.epoch = [], []

        self.ax2 = self.fig.add_subplot(1, 2, 2)
        self.ax2.set_title('The fitted curve')
        self.ax2.set_xlabel('Year')
        self.ax2.set_ylabel('Price')


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
        self.plot()
        plt.ion()
        for e in range(epochs):
            pred = self.sigmoid(np.matmul(self.x, self.weight))
            loss, grad = self.loss_fn(self.x, self.y, pred)
            self.weight -= learning_rate * grad

            # 以下为画图代码
            plt.cla()
            self.epoch.append(e)
            self.loss.append(loss)
            self.ax1.plot(self.epoch, self.loss, 'r')

            exam_1, exam_2 = self.x[:, 1], self.x[:, 2]
            x1, x2 = np.where(self.y == 0)[0], np.where(self.y == 1)[0]
            self.ax2.scatter(exam_1[x1], exam_2[x1], c='b', marker='*')
            self.ax2.scatter(exam_1[x2], exam_2[x2], c='b', marker='o')

            x_min, x_max = np.min(exam_1), np.max(exam_1)
            y_min = -(self.weight[0] + self.weight[1] * x_min) / self.weight[2]
            y_max = -(self.weight[0] + self.weight[1] * x_max) / self.weight[2]
            self.ax2.plot([x_min, x_max], [y_min, y_max], 'r')
            plt.pause(0.01)
        plt.ioff()
        plt.show()


    def stochastic_gradient_descent(self, learning_rate, epochs):
        orders = list(range(self.x.shape[0]))
        random.seed(2020)
        random.shuffle(orders)
        self.plot()
        plt.ion()
        i = 1
        for e in range(epochs):
            for index in orders:
                pred = self.sigmoid(np.matmul(self.x[index], self.weight))
                _, grad = self.loss_fn(self.x[index].reshape(1, -1), self.y[index].reshape(-1, 1), pred.reshape(-1, 1))
                self.weight -= learning_rate * grad

                # 以下为画图代码
                plt.cla()
                pred = self.sigmoid(np.matmul(self.x, self.weight))
                loss, _ = self.loss_fn(self.x, self.y, pred)
                self.epoch.append(i)
                self.loss.append(loss)
                self.ax1.plot(self.epoch, self.loss, 'r')

                exam_1, exam_2 = self.x[:, 1], self.x[:, 2]
                x1, x2 = np.where(self.y == 0)[0], np.where(self.y == 1)[0]
                self.ax2.scatter(exam_1[x1], exam_2[x1], c='b', marker='*')
                self.ax2.scatter(exam_1[x2], exam_2[x2], c='b', marker='o')

                x_min, x_max = np.min(exam_1), np.max(exam_1)
                y_min = -(self.weight[0] + self.weight[1] * x_min) / self.weight[2]
                y_max = -(self.weight[0] + self.weight[1] * x_max) / self.weight[2]
                self.ax2.plot([x_min, x_max], [y_min, y_max], 'r')
                i += 1
                plt.pause(0.01)
        plt.ioff()
        plt.show()

    def newton_method(self, epochs):
        self.plot()
        plt.ion()
        for e in range(epochs):
            pred = self.sigmoid(np.matmul(self.x, self.weight)).reshape(-1, 1)
            loss, grad = self.loss_fn(self.x, self.y, pred)
            H = np.matmul(self.x.T, pred * (1 - pred) * self.x)
            self.weight -= np.matmul(np.linalg.inv(H), grad)

            # 以下为画图代码
            plt.cla()
            self.epoch.append(e)
            self.loss.append(loss)
            self.ax1.plot(self.epoch, self.loss, 'r')

            exam_1, exam_2 = self.x[:, 1], self.x[:, 2]
            x1, x2 = np.where(self.y == 0)[0], np.where(self.y == 1)[0]
            self.ax2.scatter(exam_1[x1], exam_2[x1], c='b', marker='*')
            self.ax2.scatter(exam_1[x2], exam_2[x2], c='b', marker='o')

            x_min, x_max = np.min(exam_1), np.max(exam_1)
            y_min = -(self.weight[0] + self.weight[1] * x_min) / self.weight[2]
            y_max = -(self.weight[0] + self.weight[1] * x_max) / self.weight[2]
            self.ax2.plot([x_min, x_max], [y_min, y_max], 'r')
            plt.pause(0.01)
        plt.ioff()
        plt.show()

    def plot(self):
        self.fig = plt.figure(num=1, figsize=(15, 8),dpi=80)
        self.ax1 = self.fig.add_subplot(1, 2, 1)

        self.ax1.set_title('The loss of Logistic_Regression')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.loss, self.epoch = [], []

        self.ax2 = self.fig.add_subplot(1, 2, 2)
        self.ax2.set_title('The classification')
        self.ax2.set_xlabel('exam_1')
        self.ax2.set_ylabel('exam_2')
        

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
        self.plot()
        plt.ion()
        for e in range(epochs):
            pred = self.softmax(np.matmul(self.x, self.weight))
            loss, grad = self.loss_fn(self.x, self.y, pred)
            self.weight -= learning_rate * grad
            
            # 以下为画图代码
            plt.cla()
            self.epoch.append(e)
            self.loss.append(loss)
            self.ax1.plot(self.epoch, self.loss, 'r')

            x_1, x_2 = self.x[:, 1], self.x[:, 2]
            index_1, index_2, index_3 = np.where(self.y[:, 0] == 1)[0], np.where(self.y[:, 1] == 1)[0], np.where(self.y[:, 2] == 1)[0]
            self.ax2.scatter(x_1[index_1], x_2[index_1], c='r', marker='*', facecolors='none')
            self.ax2.scatter(x_1[index_2], x_2[index_2], c='g', marker='o', facecolors='none')
            self.ax2.scatter(x_1[index_3], x_2[index_3], c='b', marker='x', facecolors='none')

            x1_min, x2_min = np.min(self.x, axis=0)[1: ] - 0.5
            x1_max, x2_max = np.max(self.x, axis=0)[1: ] + 0.5
            _x, _y = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
            pred_x = np.c_[np.zeros((_x.size, )), _x.flatten(), _y.flatten()]
            pred_y = np.argmax(self.softmax(np.matmul(pred_x, self.weight)), axis=1).reshape(_x.shape)

            cmp = ListedColormap(colors=['yellow', 'green', 'red'])
            self.ax2.contourf(_x, _y, pred_y, cmap=cmp, alpha=0.5)
            plt.pause(0.01)
        plt.ioff()
        plt.show() 
        

    def stochastic_gradient_descent(self, learning_rate, epochs):
        orders = list(range(self.x.shape[0]))
        random.seed(2020)
        random.shuffle(orders)
        i = 1
        self.plot()
        plt.ion()
        for e in range(epochs):
            for index in orders:
                pred = np.matmul(self.x[index], self.weight)
                pred = (np.exp(pred) / np.sum(np.exp(pred)))
                _, grad = self.loss_fn(self.x[index].reshape(1, -1), self.y[index].reshape(1, -1), pred.reshape(1, -1))
                self.weight -= learning_rate * grad
            
                # 以下为画图代码
                plt.cla()
                pred = np.matmul(self.x, self.weight)
                pred = (np.exp(pred) / np.sum(np.exp(pred)))
                loss, _ = self.loss_fn(self.x, self.y, pred)
                self.epoch.append(i)
                self.loss.append(loss)
                self.ax1.plot(self.epoch, self.loss, 'r')

                x_1, x_2 = self.x[:, 1], self.x[:, 2]
                index_1, index_2, index_3 = np.where(self.y[:, 0] == 1)[0], np.where(self.y[:, 1] == 1)[0], np.where(self.y[:, 2] == 1)[0]
                self.ax2.scatter(x_1[index_1], x_2[index_1], c='r', marker='*', facecolors='none')
                self.ax2.scatter(x_1[index_2], x_2[index_2], c='g', marker='o', facecolors='none')
                self.ax2.scatter(x_1[index_3], x_2[index_3], c='b', marker='x', facecolors='none')

                x1_min, x2_min = np.min(self.x, axis=0)[1: ] - 0.5
                x1_max, x2_max = np.max(self.x, axis=0)[1: ] + 0.5
                _x, _y = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
                pred_x = np.c_[np.zeros((_x.size, )), _x.flatten(), _y.flatten()]
                pred_y = np.argmax(self.softmax(np.matmul(pred_x, self.weight)), axis=1).reshape(_x.shape)

                cmp = ListedColormap(colors=['yellow', 'green', 'red'])
                self.ax2.contourf(_x, _y, pred_y, cmap=cmp, alpha=0.5)
                i += 1
                plt.pause(0.01)
        plt.ioff()
        plt.show() 

    def plot(self):
        self.fig = plt.figure(num=1, figsize=(15, 8),dpi=80)
        self.ax1 = self.fig.add_subplot(1, 2, 1)

        self.ax1.set_title('The loss of Softmax_Regression')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.loss, self.epoch = [], []

        self.ax2 = self.fig.add_subplot(1, 2, 2)
        self.ax2.set_title('The classification')
        self.ax2.set_xlabel('x_1')
        self.ax2.set_ylabel('x_2')


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
        i = 1
        self.plot()
        plt.ion()
        for e in range(epochs):
            for index in orders:
                pred = self.hypothesis(np.matmul(self.x[index], self.weight))
                _, grad = self.loss_fn(self.x[index].reshape(1, -1), self.y[index].reshape(1, -1), pred.reshape(1, -1))
                self.weight -= learning_rate * grad

                # 以下为画图代码
                plt.cla()
                pred = self.hypothesis(np.matmul(self.x, self.weight))
                loss, _ = self.loss_fn(self.x, self.y, pred)
                self.epoch.append(i)
                self.loss.append(loss)
                self.ax1.plot(self.epoch, self.loss, 'r')

                exam_1, exam_2 = self.x[:, 1], self.x[:, 2]
                x1, x2 = np.where(self.y == 0)[0], np.where(self.y == 1)[0]
                self.ax2.scatter(exam_1[x1], exam_2[x1], c='b', marker='*')
                self.ax2.scatter(exam_1[x2], exam_2[x2], c='b', marker='o')

                x_min, x_max = np.min(exam_1), np.max(exam_1)
                y_min = -(self.weight[0] + self.weight[1] * x_min) / self.weight[2]
                y_max = -(self.weight[0] + self.weight[1] * x_max) / self.weight[2]
                self.ax2.plot([x_min, x_max], [y_min, y_max], 'r')
                i += 1
                plt.pause(0.01)
        plt.ioff()
        plt.show() 
    
    def plot(self):
        self.fig = plt.figure(num=1, figsize=(15, 8),dpi=80)
        self.ax1 = self.fig.add_subplot(1, 2, 1)

        self.ax1.set_title('The loss of Perceptron')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.loss, self.epoch = [], []

        self.ax2 = self.fig.add_subplot(1, 2, 2)
        self.ax2.set_title('The classification')
        self.ax2.set_xlabel('exam_1')
        self.ax2.set_ylabel('exam_2')
        


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
        self.plot()
        plt.ion()
        i = 1
        for e in range(epochs):
            for index in orders:
                pred = np.matmul(self.x[index], self.weight)
                _, grad = self.loss_fn(self.x[index].reshape(1, -1), self.y[index].reshape(1, -1), pred.reshape(1, -1))
                self.weight -= learning_rate * grad

                # 以下为画图代码
                plt.cla()
                pred = np.matmul(self.x, self.weight)
                loss, _ = self.loss_fn(self.x, self.y, pred)
                self.epoch.append(i)
                self.loss.append(loss)
                self.ax1.plot(self.epoch, self.loss, 'r')

                x_1, x_2 = self.x[:, 1], self.x[:, 2]
                index_1, index_2, index_3 = np.where(self.y[:, 0] == 1)[0], np.where(self.y[:, 1] == 1)[0], np.where(self.y[:, 2] == 1)[0]
                self.ax2.scatter(x_1[index_1], x_2[index_1], c='r', marker='*', facecolors='none')
                self.ax2.scatter(x_1[index_2], x_2[index_2], c='g', marker='o', facecolors='none')
                self.ax2.scatter(x_1[index_3], x_2[index_3], c='b', marker='x', facecolors='none')

                x1_min, x2_min = np.min(self.x, axis=0)[1: ] - 0.5
                x1_max, x2_max = np.max(self.x, axis=0)[1: ] + 0.5
                _x, _y = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
                pred_x = np.c_[np.zeros((_x.size, )), _x.flatten(), _y.flatten()]
                pred_y = np.argmax(np.matmul(pred_x, self.weight), axis=1).reshape(_x.shape)

                cmp = ListedColormap(colors=['yellow', 'green', 'red'])
                self.ax2.contourf(_x, _y, pred_y, cmap=cmp, alpha=0.5)
                i += 1
                plt.pause(0.01)
        plt.ioff()
        plt.show() 

    def plot(self):
        self.fig = plt.figure(num=1, figsize=(15, 8),dpi=80)
        self.ax1 = self.fig.add_subplot(1, 2, 1)

        self.ax1.set_title('The loss of Multi_Class_Perceptron')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.loss, self.epoch = [], []

        self.ax2 = self.fig.add_subplot(1, 2, 2)
        self.ax2.set_title('The classification')
        self.ax2.set_xlabel('x_1')
        self.ax2.set_ylabel('x_2')


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
        self.plot()
        plt.ion()
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
            
            # 以下为画图代码
            plt.cla()
            pred_hidden = self.sigmoid(np.matmul(self.x, self.w1) + self.b1)
            pred = self.sigmoid(np.matmul(pred_hidden, self.w2) + self.b2)
            loss, _, _, _, _ = self.loss_fn(self.x, self.y, pred_hidden, pred)
            self.epoch.append(e)
            self.loss.append(loss)
            self.ax1.plot(self.epoch, self.loss, 'r')

            x_1, x_2 = self.x[:, 1], self.x[:, 2]
            index_1, index_2, index_3 = np.where(self.y[:, 0] == 1)[0], np.where(self.y[:, 1] == 1)[0], np.where(self.y[:, 2] == 1)[0]
            self.ax2.scatter(x_1[index_1], x_2[index_1], c='r', marker='*', facecolors='none')
            self.ax2.scatter(x_1[index_2], x_2[index_2], c='g', marker='o', facecolors='none')
            self.ax2.scatter(x_1[index_3], x_2[index_3], c='b', marker='x', facecolors='none')

            x1_min, x2_min = np.min(self.x, axis=0)[1: ] - 0.5
            x1_max, x2_max = np.max(self.x, axis=0)[1: ] + 0.5
            _x, _y = np.meshgrid(np.arange(x1_min, x1_max, 0.01), np.arange(x2_min, x2_max, 0.01))
            pred_x = np.c_[np.zeros((_x.size, )), _x.flatten(), _y.flatten()]
            hidden = self.sigmoid(np.matmul(pred_x, self.w1) + self.b1)
            pred_y = np.argmax(self.sigmoid(np.matmul(hidden, self.w2) + self.b2), axis=1).reshape(_x.shape)

            cmp = ListedColormap(colors=['yellow', 'green', 'red'])
            self.ax2.contourf(_x, _y, pred_y, cmap=cmp, alpha=0.5)
            plt.pause(0.01)
        plt.ioff()
        plt.show()


    def plot(self):
        self.fig = plt.figure(num=1, figsize=(15, 8),dpi=80)
        self.ax1 = self.fig.add_subplot(1, 2, 1)

        self.ax1.set_title('The loss of Artifical_Neural_Network')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.loss, self.epoch = [], []

        self.ax2 = self.fig.add_subplot(1, 2, 2)
        self.ax2.set_title('The classification')
        self.ax2.set_xlabel('x_1')
        self.ax2.set_ylabel('x_2')


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


class Naive_Bayes():
    def __init__(self, input_x, input_y):
        self.x = input_x
        self.y = input_y
        self.class_num = np.max(self.y) + 1
        self.doc_num = [len(docs) for docs in self.x]
        
        vocabulary = []
        for docs in self.x:
            for doc in docs:
                vocabulary.extend(doc)
            
        self.vocabulary = set(vocabulary)
        self.v = len(self.vocabulary)

    def calculate_accuracy(self, pred, true):
        pred = np.argmax(pred, axis=1)
        return np.sum(pred == true) / true.shape[0]

    def multinomial(self, input_x, input_y):
        prob_words = {}
        for i in range(len(self.x)):
            prob_words[i] = {}
            docs = self.x[i]
            words_count = 0
            for doc in docs:
                words_count += len(doc)
                for token in doc:
                    if token not in prob_words[i]:
                        prob_words[i][token] = 1
                    else:
                        prob_words[i][token] += 1
            for key in prob_words[i]:
                prob_words[i][key] += 1
                prob_words[i][key] /= (words_count + self.v)

        prob_class = (np.array(self.doc_num) + 1) / (np.sum(self.doc_num) + self.class_num)

        x, y = [], []
        for i in range(len(input_y)):
            x.extend(input_x[i])
            y.extend([input_y[i]] * len(input_x[i]))

        preds = []
        for doc in x:
            pred = []
            for i in range(self.class_num):
                p = np.log(prob_class[i])
                for token in doc:
                    if token in prob_words[i]:
                        p += np.log(prob_words[i][token])
                    else:
                        p += np.log(1 / (words_count + self.v))
                pred.append(p)
            preds.append(pred)
        accuracy = self.calculate_accuracy(np.asarray(preds), np.asarray(y))
        print(accuracy)

    def bernoulli(self, input_x, input_y):
        prob_words = {}
        for i in range(len(self.x)):
            prob_words[i] = {}
            docs = self.x[i]
            for doc in docs:
                doc = set(doc)
                for token in doc:
                    if token not in prob_words[i]:
                        prob_words[i][token] = 1
                    else:
                        prob_words[i][token] += 1

            exist_words = set(prob_words[i])
            for token in self.vocabulary:
                if token not in exist_words:
                    prob_words[i][token] = 0

            for key in prob_words[i]:
                prob_words[i][key] += 1
                prob_words[i][key] /= (self.doc_num[i] + 2)

        prob_class = (np.array(self.doc_num) + 1) / (np.sum(self.doc_num) + self.class_num)

        x, y = [], []
        for i in range(len(input_y)):
            x.extend(input_x[i])
            y.extend([input_y[i]] * len(input_x[i]))

        preds = []
        for doc in x:
            doc = set(doc)
            pred = []
            for i in range(self.class_num):
                p = np.log(prob_class[i])
                for token in self.vocabulary:
                    if token in doc:
                        p += np.log(prob_words[i][token])
                    else :
                        p += np.log(1 - prob_words[i][token])
                pred.append(p)
            preds.append(pred)

        accuracy = self.calculate_accuracy(np.asarray(preds), np.asarray(y))
        print(accuracy)
