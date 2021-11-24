import numpy as np
from scores import Scores
import time

def logLike(x, mu, sigma, pi):
    ctr = x - mu
    return 2*np.log(pi) - np.sum(ctr*(ctr@np.transpose(np.linalg.inv(sigma))), axis=1) - np.log(np.linalg.det(sigma))

class QDA:

    def __init__(self):
        self.sigma1 = np.empty((3, 3))
        self.sigma0 = np.empty((3, 3))
        self.mu1 = 0
        self.mu0 = 0
        self.pi1 = 0
        self.pi0 = 0
        self.trained = False

    def __repr__(self):
        return f'Modèle entraîné : {self.trained}'

    def train(self, trainSet):
        mask1 = trainSet[:, 3] == 1
        l1 = len(mask1)
        mask0 = trainSet[:, 3] == 0
        l0 = len(mask0)
        l = len(trainSet)
        self.sigma1 = np.cov(trainSet[mask1, :3].T)
        self.sigma0 = np.cov(trainSet[mask0, :3].T)
        self.mu1 = np.mean(trainSet[mask1, :3], axis=0)
        self.mu0 = np.mean(trainSet[mask0, :3], axis=0)
        self.pi1 = l1/l
        self.pi0 = l0/l
        self.trained = True

    def predict(self, X):
        return logLike(X, self.mu1, self.sigma1, self.pi1) > logLike(X, self.mu0, self.sigma0, self.pi0)

    def performance(self, testSet):
        start = time.time()
        pred = self.predict(testSet[:, :3])
        end = time.time()
        sc = Scores(testSet[:, 3], pred)
        print(sc)
        return end - start