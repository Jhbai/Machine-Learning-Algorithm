class FDA_plot:
    def __init__(self, F, X1, X2):
        self.W = F.__coef__()
        xx = np.linspace(np.min(X1), np.max(X1), 200)
        yy = np.linspace(np.min(X2), np.max(X2), 200)
        yy = (W[0]*xx  + W[1]*yy)/W[1]
        self.X1 = X1
        self.X2 = X2
        self.xx = xx
        self.yy = yy
    def plot(self):
        plt.figure(figsize = (8, 6))
        plt.scatter(self.X1[:,0], self.X1[:,1], color = 'orange', label = 'label 0')
        plt.scatter(self.X2[:,0], self.X2[:,1], color = 'blue', label = 'label 1')
        plt.plot(self.xx, self.yy, color = 'red', label = 'LDA')
        plt.legend()
        plt.show()
