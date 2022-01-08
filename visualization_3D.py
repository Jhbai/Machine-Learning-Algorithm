import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Plot_3D:
    def __init__(self, data):
        self.data = data
    def plot(self):
        plt.figure(figsize = (10, 8))
        ax = plt.gca(projection='3d')
        X = self.data[self.data.columns[0]]
        Y = self.data[self.data.columns[1]]
        Z = self.data[self.data.columns[2]]
        ax.scatter(X, Y, Z, c = Z, marker = 'o')
        cols = self.data.columns
        ax.set_xlabel(cols[0])
        ax.set_ylabel(cols[1])
        ax.set_zlabel(cols[2])
        plt.show()
