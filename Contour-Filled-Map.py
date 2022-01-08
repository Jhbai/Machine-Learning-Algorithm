class classify_graph:
    def __init__(self, model, X, Class):
        self.xx = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 200)
        self.yy = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 200)
        self.xx, self.yy = np.meshgrid(self.xx, self.yy)
        self.xxx = self.xx.flatten()
        self.yyy = self.yy.flatten()
        self.model = model
        self.Class = Class
    def plot(self):
        zz = list()
        for i in range(self.xxx.shape[0]):
            x = np.array([self.xxx[i], self.yyy[i]]).reshape((1, 2))
            temp = self.model.predict(x)
            zz.append(temp)
        zz = np.array(zz).reshape(self.xx.shape)
        plt.figure(figsize = (8, 6))
        plt.contourf(self.xx, self.yy, zz)
        for obj in list(set(self.Class)):
            I = np.argwhere(self.Class == obj)
            plt.scatter(X[I, 0], X[I, 1])
        plt.show()
