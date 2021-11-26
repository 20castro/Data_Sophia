import numpy as np
from scores import Scores

class GaussianKernel:

    def __init__(self):
        self.class1 = None
        self.class0 = None
        self.n1 = 0
        self.n0 = 0

    def __repr__(self) -> str:
        return f'Modèle entraîné : {self.trained}'

    def train(self, trainSet):
        mask1 = trainSet[:, 3] == 1
        mask0 = trainSet[:, 3] == 0
        self.class1 = trainSet[mask1, :3]
        self.class0 = trainSet[mask0, :3]
        self.n1 = np.power(np.linalg.norm(self.class1, axis=1), 2).reshape((1, self.class1.shape[0]))
        self.n0 = np.power(np.linalg.norm(self.class0, axis=1), 2).reshape((1, self.class0.shape[0]))
        self.trained = True

    def predict(self, X):

        ## Version générale vectorisée (avec self.K en argument de __init__, le noyau)
        ## Très très lent
        
        # D1 = X.reshape((s[0], 1, s[1])) - self.class1
        # D0 = X.reshape((s[0], 1, s[1])) - self.class0
        # N1 = self.K(D1).sum(axis=1)
        # N0 = self.K(D0).sum(axis=1)

        ## Version non vectorisée (avec self.K en argument de __init__, le noyau)
        ## Très lent

        # N1 = np.zeros(s[0])
        # N0 = np.zeros(s[0])
        # for k, px in enumerate(X):
        #     N1[k] += self.K(px - self.class1).sum()
        #     N0[k] += self.K(px - self.class0).sum()

        ## Dans les deux cas correspondant, on prendrait pour self.K :
        # def gaussian(u):
        #     return np.exp(-.5*np.power(np.linalg.norm(u, axis=len(u.shape) - 1), 2))/np.sqrt(2*np.pi)
        
        ## Version vectorisée (cas particulier adopté : gaussien mais généralisable facilement)
        ## On décompose le calcul de la norme et on ne prend que les valeurs non constantes (on enlève ||x||^2)

        N1 = np.exp(- X@np.transpose(self.class1) - self.n1).sum(axis=1)
        N0 = np.exp(- X@np.transpose(self.class0) - self.n0).sum(axis=1)

        return N1 > N0 # équivalent à N1/(N0 + N1) > 1/2

    def performance(self, testSet):
        sc = Scores(testSet[:, 3], self.predict(testSet[:, :3]))
        print(sc)
        return sc