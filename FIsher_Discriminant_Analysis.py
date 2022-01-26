class FDA:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def fit(self):
        W = np.random.normal(0, 1, size = (2, ))
        y1 = list(set(self.y))[0]
        y2 = list(set(self.y))[1]
        I = np.argwhere(self.y == y1).reshape((-1, ))
        II = np.argwhere(self.y == y2).reshape((-1, ))
        COV1 = np.cov(self.X[I].T)
        COV2 = np.cov(self.X[II].T)
        COV = (1/(len(I) + len(II)))*((len(I)*COV1) + (len(II)*COV2))
        M1 = np.mean(self.X[I], axis = 0).reshape((-1, 1))
        M2 = np.mean(self.X[II], axis = 0).reshape((-1, 1))
        #print('---Start Training---')
        J = W.T@(M1 - M2)*(M1 - M2).T@W/(W.T@COV@W)
        #print('init  J:', J)
        for epoch in range(1000):
            mu = (M1 - M2)@(M1 - M2).T
            A = mu@W/(W.T@mu@W) - COV@W/(W.T@COV@W)
            GRAD = A*2*J
            W += 0.001*GRAD
            J_temp = W.T@(M1 - M2)*(M1 - M2).T@W/(W.T@COV@W)
            #print('epoch '+str(epoch)+':', J_temp)
            if np.abs(J_temp - J) < 0.00000001:
                break
            else:
                J = J_temp
        self.W = W
        
    def transform(self, data):
        self.result = (self.W.T@data.T)
        return self.result
    
    def __coef__(self):
        return self.W
