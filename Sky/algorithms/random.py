import numpy as np
from scores import Scores
from sklearn.dummy import DummyClassifier
from time import perf_counter

class RandomClassifier:

    def __init__(self):
        self.rate = 0.
        self.trained = False
        self.time = {}

        self.skPredictor = DummyClassifier(strategy='stratified')
        self.sktime = {}

    def __repr__(self) -> str:
        if self.trained:
            return f"Proportion positive dans le jeu d'entraînement : {self.rate}"
        else:
            return 'Modèle non entraîné'

    def train(self, trainSet):

        start = perf_counter()
        l = len(trainSet)
        self.rate = np.count_nonzero(trainSet[:, 3])/l
        end = perf_counter()

        self.skPredictor.fit(trainSet[:, :3], trainSet[:, 3])
        skend = perf_counter()

        self.time['Train'] = end - start
        self.sktime['Sklearn train'] = skend - end
        self.trained = True

    def predict(self, X):
        return np.random.rand(X.shape[0]) < self.rate

    def performance(self, testSet, training_rate):

        start = perf_counter()
        pred = self.predict(testSet[:, :3])
        end = perf_counter()
        skpred = self.skPredictor.predict(testSet[:, :3])
        skend = perf_counter()

        self.time['Fitting'] = end - start
        self.sktime['Sklearn fitting'] = skend - end

        sc = Scores(testSet[:, 3], pred, 'DUMMY', training_rate)
        sc.addTimes(self.time, self.sktime)
        sc.addSklearnMetrics(skpred)
        print(sc)
        return sc