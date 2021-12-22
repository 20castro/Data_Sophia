import numpy as np
from typing import Optional

from subtree import LearningNode, impurete

class Root(LearningNode):

    def __init__(self):
        super().__init__(0, 1)

    def __pruning(self):
        alpha = [0]
        R = [self.subR]
        n = 2
        while not 1 in self.nodeMinInSubtree:
            self.__round(n, [f'{nodeMin:b}'[1:] for nodeMin in self.nodeMinInSubtree])
            alpha.append(self.gMinInSubtree)
            R.append(self.subR)
            n += 1

    def fit(self, X, y, variableType: Optional[np.ndarray]):

        '''
        X : array of feature of shape (n, p) with n the number of samples, p the number of features
        it is assumed that X elements are numbers (int, floats)
        y : labels
        variableType : boolean array of size (p,) indicating if each feature is categorical or not
        (no one hot encoding is necessary)
        if None, it is assumed that none of them are categorical
        '''

        assert (y == 0).size > 0 and (y == 1).size > 0, 'Uncomplete training set'

        self.__i = impurete(y)
        self.subproportion = 1
        if not (variableType is None):
            self.__variableType  = variableType
        else:
            self.__variableType = np.zeros(X.shape[1], dtype=np.bool)
        self.grow(X, y)
        self.__pruning()

    def predict(self, X):
        return self.__call__(X)