import numpy as np
from math import sqrt

def display(title: str, value, totalLength=61):
    str_value = str(value)
    lt = len(title)
    lv = len(str_value)
    add = totalLength - lt - lv - 3
    if add < 0:
        raise ValueError('Too short total length')
    return title + ' : ' + add*' ' + str_value + '\n'


class Scores:

    def __init__(self, labels, predicted_labels):
        self.labels = labels.astype('bool')
        self.predicted_labels = predicted_labels.astype('bool')

        self.L = len(labels)
        self.TP = np.count_nonzero(labels[predicted_labels])
        self.TN = np.count_nonzero(np.logical_not(labels[np.logical_not(predicted_labels)]))
        self.FP = np.count_nonzero(np.logical_not(labels[predicted_labels]))
        self.FN = np.count_nonzero(labels[np.logical_not(predicted_labels)])
        self.P = self.TP + self.FN
        self.N = self.TN + self.FP

    ## Scores

    def recall(self):
        return self.TP/(self.TP + self.FN)

    def specificity(self):
        return self.TN/(self.TN + self.FP)

    def FalsePositiveRate(self):
        return self.FP/(self.TN + self.FP)

    def precision(self):
        return self.TP/(self.TP + self.FP)

    def FOV(self):
        return self.FP/(self.FP + self.TN)

    def error(self):
        return (self.FP + self.FN)/self.L

    def accuracy(self):
        return (self.TP + self.TN)/self.L

    def MCC(self):
        return (self.TP*self.TN - self.FP*self.FN)/sqrt((self.TP + self.FP)*(self.TP + self.FN)*(self.TN + self.FP)*(self.TN + self.FN))

    def FScore(self, beta):
        ppv = self.precision()
        tpr = self.recall()
        return (1 + beta**2)*ppv*tpr/(ppv*beta**2 + tpr)

    def kappa(self):
        P0 = self.P/(self.P + self.N)
        P1 = self.N/(self.P + self.N)
        tpr = self.recall()
        fpr = self.FalsePositiveRate()
        err = self.error()
        fact = 2*P0*P1*(tpr - fpr)
        return fact/(fact + err)

    def __repr__(self):
        sep = 61*'_'
        # Version longue
        vlong = sep + '\n' + \
                sep + '\n' + \
                display('F1-score', self.FScore(1)) + \
                sep + '\n' + \
                display('Kappa', self.kappa()) + \
                sep + '\n' + \
                display('Recall', self.recall()) + \
                sep + '\n' + \
                display('Specificity', self.specificity()) + \
                sep + '\n' + \
                display('False positive rate', self.FalsePositiveRate()) + \
                sep + '\n' + \
                display('Precision', self.precision()) + \
                sep + '\n' + \
                display('False omission rate', self.FOV()) + \
                sep + '\n' + \
                display('Error', self.error()) + \
                sep + '\n' + \
                display('Accuracy', self.accuracy()) + \
                sep + '\n' + \
                display('Corrélation de Matthews', self.MCC()) + \
                sep
        # Version courte
        vshort = sep + '\n' + \
                sep + '\n' + \
                display('F1-score', self.FScore(1)) + \
                sep + '\n' + \
                display('Recall', self.recall()) + \
                sep + '\n' + \
                display('Precision', self.precision()) + \
                sep + '\n' + \
                display('Accuracy', self.accuracy()) + \
                sep
        return vshort