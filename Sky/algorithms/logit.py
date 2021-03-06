import numpy as np
from scores import Scores
from sklearn.linear_model import LogisticRegression
from time import perf_counter

def log_likelihood_derivatives(beta: np.ndarray, u1: np.ndarray, u0: np.ndarray):

    e1 = np.exp(u1@beta)
    e0 = np.exp(u0@beta)
    f1 = e1/(1. + e1)
    f1.resize(f1.shape[0], 1)
    f0 = e0/(1. + e0)
    f0.resize(f0.shape[0], 1)
    deriv1 = np.sum(u1, axis=0) - np.sum(f1*u1, axis=0) - np.sum(f0*u0, axis=0)
    
    g1 = (e1/np.power(1. + e1, 2)).reshape((e1.shape[0], 1))*u1
    g0 = (e0/np.power(1. + e0, 2)).reshape((e0.shape[0], 1))*u0
    s1 = g1.shape
    s0 = g0.shape
    m1 = np.matmul(g1.reshape((s1[0], s1[1], 1)), u1.reshape((s1[0], 1, s1[1]))).sum(axis=0)
    m0 = np.matmul(g0.reshape((s0[0], s0[1], 1)), u0.reshape((s0[0], 1, s0[1]))).sum(axis=0)
    deriv2 = - m1 - m0

    return deriv1, deriv2

class Logit:

    def __init__(self):
        self.beta = np.zeros(4) # intialisation pour l'optimisation
        self.step = None
        self.trained = False
        self.time = {}

        self.skPredictor = LogisticRegression(penalty='none')
        self.sktime = {}
        self.sktrained = False

    def __repr__(self) -> str:
        if self.trained:
            return f'Modèle entraîné en {self.step} étapes avec le vecteur {self.beta}'
        else:
            return f'Modèle non entraîné'

    def train(self, trainSet):

        start = perf_counter()
        s = trainSet.shape
        u = np.ones(s)
        u[:, 1:] = trainSet[:, :3] # première colonne de 1, le reste rempli par les pixels
        mask1 = trainSet[:, 3] == 1
        mask0 = trainSet[:, 3] == 0

        # Optimisation de la log-vraisemblance

        step = 0
        N = 1000
        eps = 10**-4
        u1 = u[mask1]
        u0 = u[mask0]
        while True:
            if step > N:
                raise RuntimeError(f'Pas de convergence en {N} étape')
            d1, d2 = log_likelihood_derivatives(self.beta, u1, u0)
            inc = np.linalg.inv(d2)@d1
            if np.linalg.norm(inc) < eps:
                break
            else:
                self.beta -= inc
            step += 1
        
        end = perf_counter()

        self.trained = True
        self.step = step
        self.time['Train'] = end - start

    def sktrain(self, trainSet):
        start = perf_counter()
        self.skPredictor.fit(trainSet[:, :3], trainSet[:, 3])
        end = perf_counter()
        self.sktime['Sklearn train'] = end - start
        self.sktrained = True

    def predict(self, X):
        return self.beta[0] + X@self.beta[1:4] > 0

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

            sc = Scores(testSet[:, 3], pred, 'LOGIT', training_rate)
            sc.addTimes(self.time, self.sktime)
            sc.addSklearnMetrics(skpred)
            
        else:

            self.time['Fitting'] = end - start
            sc = Scores(testSet[:, 3], pred, 'LOGIT', training_rate)

        print(sc)
        return sc