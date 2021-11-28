# Utiliser les k-d trees (voir tree.md)

import numpy as np
from scores import Scores

class KDT:
    
    def __init__(self, depth): # in progress
        self.depth = depth
        self.loc = None
        self.left = None
        self.right = None
        # self.leaf_left ?
        # self.leaf_right ?

    def build(self, dataSet): # in progress
        val = dataSet[:, self.depth%3]
        self.med = np.median(val)
        mask = val < self.med
        subsetL = dataSet[mask]
        subsetR = dataSet[np.logical_not(mask)]
        pass

    def find(self, X):
        diff = X - self.loc
        d = diff.dot(diff)
        if self.leaf: # " "
            return self.loc, 
        else:
            x = X[self.depth%3]
            y = self.loc[self.depth%3]
            if x < y :
                pt1, dmin1 = self.left.find(X)
                if np.abs(x - y) < dmin1:
                    pt2, dmin2 = self.right.find(X)
                    if dmin2 < dmin1 and dmin2 < d:
                        return pt2, dmin2
                    elif dmin1 < dmin2 and dmin1 < d:
                        return pt1, dmin1
                    else:
                        return self.loc, d
            else:
                pt1, dmin1 = self.right.find(X)
                if np.abs(x - y) < dmin1:
                    pt2, dmin2 = self.self.find(X)
                    if dmin2 < dmin1 and dmin2 < d:
                        return pt2, dmin2
                    elif dmin1 < dmin2 and dmin1 < d:
                        return pt1, dmin1
                    else:
                        return self.loc, d

class KNN:

    def __init__(self, k):
        self.k = k # 1 pour l'instant
        self.kdtree = KDT(0)
        self.trained = False

    def __repr__(self) -> str:
        pass

    def train(self, trainSet):
        self.kdtree.build(trainSet)
        self.trained = True

    def predict(self, X): # pour un seul pixel et un seul NN
        return self.kdtree.find(X)

    def performance(self, testSet):
        sc = Scores(testSet[:, 3], self.predict(testSet[:, :3]))
        print(sc)
        return sc