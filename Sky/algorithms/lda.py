import numpy as np
from scores import Scores
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from time import perf_counter

class LDA:

    def __init__(self):

        self.sigma = np.empty((3, 3))
        self.mu1 = 0
        self.mu0 = 0
        self.w = np.empty(3)
        self.c = 0

        self.time = {}
        self.trained = False

        self.skPredictor = LinearDiscriminantAnalysis()
        self.sktime = {}
        self.sktrained = False

    def __repr__(self):
        if self.trained:
            return f"Matrice de covariance : {self.sigma}\n"\
                f"Moyenne pour la classe 0 : {self.mu0} ; pour la classe 1 : {self.mu1}"
        else:
            return f'Modèle non entraîné'

    def train(self, trainSet):

        start = perf_counter()
        self.sigma = np.cov(trainSet[:, :3].T)
        self.mu1 = np.mean(trainSet[trainSet[:, 3] == 1, :3], axis=0)
        self.mu0 = np.mean(trainSet[trainSet[:, 3] == 0, :3], axis=0)
        self.w = np.linalg.inv(self.sigma)@(self.mu1 - self.mu0)
        self.c = .5*np.dot(self.w, self.mu1 + self.mu0)
        end = perf_counter()

        self.trained = True
        self.time['Train'] = end - start

    def sktrain(self, trainSet):
        start = perf_counter()
        self.skPredictor.fit(trainSet[:, :3], trainSet[:, 3])
        end = perf_counter()
        self.sktime['Sklearn train'] = end - start
        self.sktrained = True

    def predict(self, X):
        return X@self.w > self.c

    def skfit(self, testSet):
        return self.skPredictor.predict(testSet[:, :3])

    def performance(self, testSet, training_rate):

        if not self.trained:
            raise "Untrained"

        start = perf_counter()
        pred = self.predict(testSet[:, :3])
        end = perf_counter()
    
        if self.sktrained:

            skpred = self.skfit(testSet)
            skend = perf_counter()

            self.time['Fitting'] = end - start
            self.sktime['Sklearn fitting'] = skend - end

            sc = Scores(testSet[:, 3], pred, 'LDA', training_rate)
            sc.addTimes(self.time, self.sktime)
            sc.addSklearnMetrics(skpred)
            
        else:

            self.time['Fitting'] = end - start
            sc = Scores(testSet[:, 3], pred, 'LDA', training_rate)

        print(sc)
        return sc