import numpy as np
from scores import Scores

class RandomClassifier:

    def __init__(self):
        self.rate = 0.
        self.trained = False

    def __repr__(self) -> str:
        if self.trained:
            return f"Proportion positive dans le jeu d'entraînement : {self.rate}"
        else:
            return 'Modèle non entraîné'

    def train(self, trainSet):
        l = len(trainSet)
        self.rate = np.count_nonzero(trainSet[:, 3])/l
        self.trained = True

    def predict(self, X):
        return np.random.rand(X.shape[0]) < self.rate

    def performance(self, testSet):
        sc = Scores(testSet[:, 3], self.predict(testSet[:, :3]))
        print(sc)
        return sc