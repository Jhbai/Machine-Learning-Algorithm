import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class PCA:
    def __init__(self, data):
        self.Mean = np.mean(data, axis = 0)
        self.data = data - self.Mean
        self.ncol = data.shape[1]
        self.nrow = data.shape[0]
    def fit(self):
        COV = self.data.T@self.data * (1/(self.nrow - 1))
        self.values, self.vectors = np.linalg.eig(COV)
        #temp = [[i, self.values[i]] for i in range(self.ncol)]
        #temp.sort(key = lambda x: x[1], reverse = True)
        #temp = np.array(temp)
        #I = np.array(temp[:,0], dtype = int)
        #self.values = self.values[I].realm
        #self.vectors = self.vectors[I].real
        self.values = self.values.real
        self.vectors = self.vectors.real
        #del temp
    def transform(self, data2):
        self.data2 = data2 - self.Mean
        return self.data2@self.vectors
    def variance(self):
        return self.values
    def variance_ratio(self):
        return self.values/np.sum(self.values)
    def select(self):
        temp = self.variance()
        N = np.sum(temp >= 0)
        return N
