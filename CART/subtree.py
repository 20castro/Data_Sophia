from node import Node
from typing import List

import numpy as np

def impurete(y):
    Nt = y.size
    N1t = np.count_nonzero(y)
    p1t = N1t/Nt
    return p1t*(1 - p1t)

class LearningNode(Node):

    def __init__(self, depth, id):
        super().__init__(depth, id)

        # Pre training features
        self.stopCard = 5
        self.__i = None
        self.__variableType = None

        # For training features
        self.__Delta = -1
        self.__tamp = {}

        # Node post training features
        self.subproportion = None
        self.maxClass = None
        self.trainingError = None

        # Subtree post training features
        self.cardT = 0
        self.subR = None

        # Pre pruning
        self.g = None
        self.gMinInSubtree = None
        self.nodeMinInSubtree = []

        # Cross validation
        self.cutAtRound = 0

    # Tree structure

    def _set(self, i, vt, p):
        self.__i = i
        self.__variableType = vt
        self.subproportion = p

    def appendChild(self):
        self.left = LearningNode(self.depth + 1, 2*self.id)
        self.right = LearningNode(self.depth + 1, 2*self.id + 1)
        self.left._set(self.__tamp['il'], self.__variableType, self.__tamp['pl'])
        self.right._set(self.__tamp['ir'], self.__variableType, self.__tamp['pr'])

    def cut(self):
        self.isLeaf = True
        self.left = None
        self.right = None

    def apply(self):
        self.feature = self._tamp['feature']
        self.value = self.__tamp['value']
        self.categorical = self.__tamp['categorical']

    def findClass(self, y):
        m = y.mean()
        if m > .5:
            self.maxClass = 1
            self.trainingError = 1. - m
        elif m == .5:
            self.maxClass = np.random.randint(0, 2)
            self.trainingError = .5
        else:
            self.maxClass = 0
            self.trainingError = m

    def actualizeInner(self):
        # we fill the function that is to minimize
        R = self.left.subR + self.right.subR
        cardT = 2 + self.left.cardT + self.right.cardT
        self.g = (self.trainingError - R)/(cardT - 1)
        if (not self.left.g is None) and (not self.right.g is None):

            if self.left.gMinInSubtree < self.right.gMinInSubtree and self.right.gMinInSubtree < self.g:
                self.gMinInSubtree = self.left.gMinInSubtree
                self.nodeMinInSubtree = self.left.nodeMinInSubtree
            elif self.left.gMinInSubtree == self.right.gMinInSubtree < self.g:
                self.gMinInSubtree = self.right.gMinInSubtree
                self.nodeMinInSubtree = self.left.nodeMinInSubtree + self.right.nodeMinInSubtree
            elif self.right.gMinInSubtree < self.left.gMinInSubtree and self.left.gMinInSubtree < self.g:
                self.gMinInSubtree = self.right.gMinInSubtree
                self.nodeMinInSubtree = self.right.nodeMinInSubtree
            else:
                self.gMinInSubtree = self.g
                self.nodeMinInSubtree = [self.id]

        elif (not self.left.g is None) and self.left.gMinInSubtree < self.g:
            self.gMinInSubtree = self.left.gMinInSubtree
            self.nodeMinInSubtree = self.left.nodeMinInSubtree

        elif (not self.right.g is None) and self.right.gMinInSubtree < self.g:
            self.gMinInSubtree = self.right.gMinInSubtree
            self.nodeMinInSubtree = self.right.nodeMinInSubtree

        else:
            self.gMinInSubtree = self.g
            self.nodeMinInSubtree = [self.id]
        
        self.cardT = cardT
        self.subR = self.subproportion*R

    def actualizeNewLeaf(self, nRound):
        self.cutAtRound = nRound
        self.g = None
        self.gMinInSubtree = None
        self.nodeMinInSubtree = None
        self.cardT = 0
        self.subR = self.subproportion*self.trainingError

    def checkUse(self):
        # never called when self is a leaf
        if self.left.isLeaf and self.right.isLeaf and self.left.maxClass == self.right.maxClass:
            # removing the useless inner nodes after growing Tmax
            self.cut()
            self.actualizeNewLeaf(1)
        else:
            self.actualizeInner()

    # Training

    def __split(self, X, y, i: int):

        Xi = X[:, i]
        values, count = np.unique(Xi, return_counts=True)

        if self.__variableType[i]:

            values = values[np.argsort(count)]
            d = values.size

            for i in range(d + 1):
                mask = np.isin(Xi, values[:i])
                antimask = np.logical_not(mask)
                potentialLeftLabels = y[mask]
                potentialRightLabels = y[antimask]
                il, ir = impurete(potentialLeftLabels), impurete(potentialRightLabels)
                Nl, Nr = potentialLeftLabels.size, potentialRightLabels.size
                pl = Nl/(Nl + Nr)
                pr = Nr/(Nl + Nr)
                Di = self.__i - pl*il - pr*ir

                if Di > self.__Delta:
                    self.__Delta = Di
                    self.__tamp['feature'] = i
                    self.__tamp['categorical'] = True
                    self.__tamp['value'] = values[:i]
                    self.__tamp['Xl'] = X[:, mask]
                    self.__tamp['Xr'] = X[:, antimask]
                    self.__tamp['yl'] = potentialLeftLabels
                    self.__tamp['yr'] = potentialRightLabels
                    self.__tamp['il'] = il
                    self.__tamp['ir'] = ir
                    self.__tamp['pl'] = pl
                    self.__tamp['pr'] = pr

        else:
            
            for v in values:
                mask = Xi < v
                antimask = np.logical_not(mask)
                potentialLeftLabels = y[mask]
                potentialRightLabels = y[antimask]
                il, ir = impurete(potentialLeftLabels), impurete(potentialRightLabels)
                Nl, Nr = potentialLeftLabels.size, potentialRightLabels.size
                pl = Nl/(Nl + Nr)
                pr = Nr/(Nl + Nr)
                Di = self.__i - pl*il - pr*ir

                if Di > self.__Delta:
                    self.__Delta = Di
                    self.__tamp['feature'] = i
                    self.__tamp['categorical'] = False
                    self.__tamp['value'] = values[:i]
                    self.__tamp['Xl'] = X[:, mask]
                    self.__tamp['Xr'] = X[:, antimask]
                    self.__tamp['yl'] = potentialLeftLabels
                    self.__tamp['yr'] = potentialRightLabels
                    self.__tamp['il'] = il
                    self.__tamp['ir'] = ir
                    self.__tamp['pl'] = pl
                    self.__tamp['pr'] = pr

    def __flow(self):
        self.apply()
        self.appendLeft()
        self.appendRight()
        self.left.grow(self.__tamp['Xl'], self.__tamp['yl'])
        self.right.grow(self.__tamp['Xr'], self.__tamp['yr'])
        self.checkUse()
        self.__tamp = {}

    def grow(self, X, y):
        self.findClass(y)
        if len(y) > self.stopCard and self.trainingError > 0 and len(self.__tamp['yl']) > 0 and len(self.__tamp['yr']) > 0:
            p = X.shape[1]
            for i in range(p):
                self.__split(X, y, i)
            self.__flow()
        else:
            # if the node is pure or if it satisfies the stop condition (N <= 5) or if one of the training data subset is empty
            # then the node is leaf
            self.isLeaf = True
            self.cardT = 0
            self.R = self.subproportion*self.trainingError

    # Pruning

    def __round(self, nRound: int, path: List[str]):
        if '' in path:
            self.actualizeNewLeaf(nRound)
        else:
            leftPaths = [p[1:] for p in path if p[0] == 0]
            rightPaths = [p[1:] for p in path if p[0] == 1]
            if len(leftPaths) > 0:
                self.left.__round(nRound, leftPaths)
            if len(rightPaths) > 0:
                self.right.__round(nRound, rightPaths)
            self.actualizeInner()