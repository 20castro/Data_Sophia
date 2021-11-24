import numpy as np
from scores import Scores
import time

class LDA:

    def __init__(self):
        self.sigma = np.empty((3, 3))
        self.mu1 = 0
        self.mu0 = 0
        self.w = np.empty(3)
        self.c = 0
        self.trained = False

    def __repr__(self):
        if self.trained:
            return f"Matrice de covariance : {self.sigma}\n"\
                f"Moyenne pour la classe 0 : {self.mu0} ; pour la classe 1 : {self.mu1}"
        else:
            return f'Modèle non entraîné'

    def train(self, trainSet):
        self.sigma = np.cov(trainSet[:, :3].T)
        self.mu1 = np.mean(trainSet[trainSet[:, 3] == 1, :3], axis=0)
        self.mu0 = np.mean(trainSet[trainSet[:, 3] == 0, :3], axis=0)
        self.w = np.linalg.inv(self.sigma)@(self.mu1 - self.mu0)
        self.c = .5*np.dot(self.w, self.mu1 + self.mu0)
        self.trained = True

    def predict(self, X):
        return X@self.w > self.c

    def performance(self, testSet):
        start = time.time()
        pred = self.predict(testSet[:, :3])
        end = time.time()
        sc = Scores(testSet[:, 3], pred)
        print(sc)
        return end - start