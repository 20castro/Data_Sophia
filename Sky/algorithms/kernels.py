import numpy as np
from scores import Scores
import time


class Kernels:

    def __init__(self, K):
        self.K = K
        self.class1 = None
        self.class0 = None

    def __repr__(self) -> str:
        return f'Modèle entraîné : {self.trained}'

    def train(self, trainSet):
        mask1 = trainSet[:, 3] == 1
        mask0 = trainSet[:, 3] == 0
        self.class1 = trainSet[mask1, :3]
        self.class0 = trainSet[mask0, :3]
        self.trained = True

    def predict(self, X): # à vectoriser (suppose ici que X est un vecteur)
        D1 = X - self.class1
        D0 = X - self.class0
        N1 = self.K(D1).sum()
        N0 = self.K(D0).sum()
        return N0 > N1 # équivalent à N0/(N0 + N1) > 1/2

    def performance(self, testSet):
        start = time.time()
        pred = self.predict(testSet[:, :3])
        end = time.time()
        sc = Scores(testSet[:, 3], pred)
        print(sc)
        return end - start