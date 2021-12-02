import numpy as np
from math import sqrt
import sklearn.metrics as metrics

def display(title: str, value, totalLength=61, end='\n'):
    str_value = str(value)
    lt = len(title)
    lv = len(str_value)
    add = totalLength - lt - lv - 3
    if add < 0:
        raise ValueError('Too short total length')
    return title + ' : ' + add*' ' + str_value + end


class Scores:

    def __init__(self, labels, predicted_labels, model_name, training_rate):
        self.labels = labels.astype('bool')
        self.predicted_labels = predicted_labels.astype('bool')
        self.name = model_name
        self.rate = training_rate

        self.L = len(labels)
        self.TP = np.count_nonzero(labels[predicted_labels])
        self.TN = np.count_nonzero(np.logical_not(labels[np.logical_not(predicted_labels)]))
        self.FP = np.count_nonzero(np.logical_not(labels[predicted_labels]))
        self.FN = np.count_nonzero(labels[np.logical_not(predicted_labels)])
        self.P = self.TP + self.FN
        self.N = self.TN + self.FP

        self.sklearnF1 = None
        self.sklearnRecall = None
        self.sklearnPrec = None
        self.sklearnAcc = None

    def addSklearnMetrics(self, ypred):
        self.sklearnF1 = metrics.f1_score(self.labels, ypred)
        self.sklearnRecall = metrics.recall_score(self.labels, ypred)
        self.sklearnPrec = metrics.precision_score(self.labels, ypred)
        self.sklearnAcc = metrics.accuracy_score(self.labels, ypred)

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
        if self.sklearnF1 is None or self.sklearnRecall is None or self.sklearnPrec is None or self.sklearnAcc is None:
            vshort = sep + '\n\n' + \
                    'Scores for model ' + self.name + 'with training rate' + str(100*self.rate) + ' %\n' + \
                    sep + '\n' + \
                    sep + '\n\n' + \
                    display('F1-score', self.FScore(1)) + \
                    sep + '\n\n' + \
                    display('Recall', self.recall()) + \
                    sep + '\n\n' + \
                    display('Precision', self.precision()) + \
                    sep + '\n\n' + \
                    display('Accuracy', self.accuracy()) + \
                    sep + '\n'
        else:
            vshort = sep + '___' + sep + '\n\n' + \
                    'Scores for model ' + self.name + ' with training rate ' + str(100*self.rate) + ' %\n' + \
                    sep + '___' + sep + '\n' + \
                    sep + '___' + sep + '\n\n' + \
                    display('F1-score', self.FScore(1), end=' | ') + display('Sklean F1-score', self.sklearnF1) + \
                    sep + ' | ' + sep + '\n\n' + \
                    display('Recall', self.recall(), end=' | ') + display('Sklean Recall', self.sklearnRecall) + \
                    sep + ' | ' + sep + '\n\n' + \
                    display('Precision', self.precision(), end=' | ') + display('Sklean Precision', self.sklearnPrec) + \
                    sep + ' | ' + sep + '\n\n' + \
                    display('Accuracy', self.accuracy(), end=' | ') + display('Sklean Accuracy', self.sklearnAcc) + \
                    sep + '___' + sep + '\n'
        return vshort