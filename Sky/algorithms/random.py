import numpy as np
from scores import Scores
from sklearn.dummy import DummyClassifier

class RandomClassifier:

    def __init__(self):
        self.rate = 0.
        self.trained = False

        self.skPredictor = DummyClassifier(strategy='stratified')

    def __repr__(self) -> str:
        if self.trained:
            return f"Proportion positive dans le jeu d'entraînement : {self.rate}"
        else:
            return 'Modèle non entraîné'

    def train(self, trainSet):
        l = len(trainSet)
        self.rate = np.count_nonzero(trainSet[:, 3])/l

        self.skPredictor.fit(trainSet[:, :3], trainSet[:, 3])

        self.trained = True

    def predict(self, X):
        return np.random.rand(X.shape[0]) < self.rate

    def performance(self, testSet, training_rate):
        sc = Scores(testSet[:, 3], self.predict(testSet[:, :3]), 'dummy', training_rate)
        sc.addSklearnMetrics(self.skPredictor.predict(testSet[:, :3]))
        print(sc)
        return sc