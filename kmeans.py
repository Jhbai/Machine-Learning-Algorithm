import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class kmean:
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.Index = [i for i in range(data.shape[0])]
    def fit(self):
        # 隨機找點作為中心
        I = np.random.choice(self.Index, size = self.k)
        self.centers = self.data[I] ## 初始化中心
        temp = np.zeros(shape = (self.k, self.data.shape[1])) ## 暫存新中心
        diff = np.ones(shape = (self.k, self.data.shape[1]))
        epoch = 1
        print('-----start train-----')
        while (np.abs(diff) > 0.000001).any():
            # 計算各點跟中心的距離
            Class = np.zeros(shape = (self.data.shape[0], self.k)) ## 距離
            classify = np.zeros(shape = (self.data.shape[0], 1)) ## 類別
            for c in range(self.k): ## 所有類別計算
                Centers = np.tile(self.centers[c], (self.data.shape[0], 1)) ## 中心拓展
                Class[:,c] = np.sqrt(np.sum(((self.data - Centers)**2), axis = 1)) ## 儲存距離於第c類
            I = np.argmin(Class, axis = 1) ## 類別結果
            for c in range(self.k):
                idx = np.argwhere(I == c) ## 找出類別c的index
                temp[c, :] = np.mean(self.data[idx], axis = 0) ## 計算該類別平均並存到暫存
            diff = temp - self.centers
            self.centers = temp
            print('Epoch', epoch, '---total distance:', np.round(np.sum(diff), 5) )
            epoch += 1
        print('-----end train-----')
        
    def predict(self, x):
        temp = np.zeros(shape = (x.shape[0], self.k))
        for c in range(self.k):
            temp[:, c] = np.sqrt(np.sum((x - self.centers[c])**2, axis = 1))
        result = np.argmin(temp, axis = 1)
        return result
