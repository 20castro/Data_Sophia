import numpy as np
from scores import Scores
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from time import perf_counter

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

        self.time = {}
        self.trained = False

        self.skPredictor = QuadraticDiscriminantAnalysis()
        self.sktime = {}
        self.sktrained = False

    def __repr__(self):
        if self.trained:
            return f'Modèle entraîné : {self.trained}'
        else:
            return f'Modèle non entraîné'

    def train(self, trainSet):

        start = perf_counter()
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
        end = perf_counter()
        
        self.time['Train'] = end - start
        self.trained = True

    def sktrain(self, trainSet):
        start = perf_counter()
        self.skPredictor.fit(trainSet[:, :3], trainSet[:, 3])
        end = perf_counter()
        self.sktime['Sklearn train'] = end - start
        self.sktrained = True

    def predict(self, X):
        return logLike(X, self.mu1, self.sigma1, self.pi1) > logLike(X, self.mu0, self.sigma0, self.pi0)

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

            sc = Scores(testSet[:, 3], pred, 'QDA', training_rate)
            sc.addTimes(self.time, self.sktime)
            sc.addSklearnMetrics(skpred)
            
        else:

            self.time['Fitting'] = end - start
            sc = Scores(testSet[:, 3], pred, 'QDA', training_rate)

        print(sc)
        return sc