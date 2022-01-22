import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Logistic_Regression:
    def __init__(self, X, y):
        self.X = X
        self.y = y.reshape((-1, 1))
    def softmax(self, data):
        up = np.ones(shape = (data.shape[0], 1))
        Data = np.concatenate([np.ones(shape = (data.shape[0], 1)), data], axis = 1)
        down = 1 + np.exp(Data@self.weight) # W 是 (D, 1)、X 是(N, D)
        return (up/down)
    def gradient(self, target, predict):
        # (D, N)*(N, 1)
        data = np.concatenate([np.ones(shape = (self.X.shape[0], 1)), self.X], axis = 1)
        return (1/target.shape[0])*(data.T@(predict - target))
    def fit(self, max_iter):
        self.weight = np.zeros(shape = (self.X.shape[1] + 1, 1))
        for epoch in range(max_iter):
            #print('Epoch', epoch,'start !!', end = '-----')
            predict = self.softmax(self.X)
            Gradient = self.gradient(self.y, predict)
            #print(predict.shape,self.y.shape)
            self.weight = self.weight + 0.01*Gradient # 更新權重
            cover = self.predict(self.X) == self.y
            acc = np.sum(cover)/self.y.shape[0]
            print('Training Acc:', acc)
    def predict(self, data):
        data = data.reshape((-1, self.X.shape[1]))
        result = self.softmax(data)
        I = np.where(result >= 0.5)
        II = np.where(result < 0.5)
        result[I] = 1
        result[II] = 0
        return result
