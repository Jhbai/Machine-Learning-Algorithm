class KNN:
    def __init__(self, X, Y, k):
        self.X = X
        self.Y = Y
        self.k = k
    def predict(self, x):
        result = list()
        for idx in range(x.shape[0]):
            distances = np.sqrt(np.sum((np.tile(x[idx], (self.X.shape[0], 1)) - self.X)**2, axis = 1))
            I = np.argsort(distances)[:self.k]
            unique, counts = np.unique(self.Y[I], return_counts = True)
            result.append(unique[0])
        return result
