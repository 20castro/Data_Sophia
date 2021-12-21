import numpy as np
from typing import Optional, Tuple
from itertools import zip_longest

class Node:

    def __init__(self, depth, id, isLeaf: bool = False):

        # Intrinsec node features
        self.depth = depth
        self.id = id
        self.isLeaf = isLeaf

        # Children
        self.left = None
        self.right = None

        # For prediction features
        self.feature = None
        self.value = None
        self.categorical = None

    # Tree structure

    def maxDepth(self):
        return self.depth if self.isLeaf else 1 + max(self.left.maxDepth(), self.right.maxDepth())

    # Drawing

    def listing(self, offset=1):
        if self.isLeaf:
            return ['o'], 1
        else:
            leftList, wl = self.left.listing(offset)
            rightList, wr = self.right.listing(offset)
            w = max(wl, wr)
            res = ['o'.center(2*w + offset, ' ')]
            for x, y in zip_longest(leftList, rightList, fillvalue=' '*w):
                res.append(x.center(w, ' ') + ' '*offset + y.center(w, ' '))
            return res, 2*w + offset

    def __repr__(self):
        res = '\n'
        for line in self.listing()[0]:
            res += line
            res += '\n'
        return res

    # Prediction

    def __call__(self, x):
        if self.isLeaf:
            return self.maxClass
        elif self.categorical:
            return self.left(x) if x[self.feature] in self.value else self.right(x)
        else:
            return self.left(x) if x[self.feature] < self.value else self.right(x)

##### Main #####

a = Node(0, 1, False)
a.left = Node(1, 2, True)
a.right = Node(1, 3, False)
a.right.left = Node(2, 6, False)
a.right.right = Node(2, 7, True)
a.right.left.right = Node(3, 12, False)
a.right.left.left = Node(3, 13, True)
a.right.left.right.right = Node(4, 24, True)
a.right.left.right.left = Node(4, 25, True)
print(a)

##### Obsolète #####


class Leaf(Node):

    def __init__(self, depth):
        super().__init__(depth)
        self.id = None
        self.trainingError = None

    def childrenCard(self):
        return 0

    def error(self):
        return self.trainingError

class Internal(Node):

    # Pruning the tree

    def childrenCard(self):
        return 2 + self.right.childrenCard() + self.right.childrenCard()

    def error(self):
        return self.left.error() + self.right.error()

    def __pruning(self):
        pass

    # Fit and predict

    def fit(self, X, y, variableType=None):

        if not (variableType is None):
            self.__variableType = variableType

        self.__grow(X, y)
        self.__pruning()