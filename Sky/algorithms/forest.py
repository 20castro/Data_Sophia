import numpy as np
from scores import Scores

def trainSubset(trainSet, N) -> np.ndarray:
    # bootstrap et subsampling
    pass

class BinaryTree:

    def __init__(self, depth, maxDepth): # il faudrait ajouter les features prises
        self.maxDepth = maxDepth
        self.depth = depth
        self.splittingFeature = None
        self.splittingValue = None
        self.left = None # Bool ou BinaryTree
        self.right = None # Bool ou BinaryTree

    def recSplit(self, subset):

        x, y = subset.shape
        kmin, smin, cmin = 0, subset[0, 0], x + 1 # le coût est majoré par x
        subL, subR, class1L, class1R, nL, nR = [], [], 0, 0, 0, 0

        for k in range(y - 1): # itération sur les features
            val = subset[:, k].reshape(x)
            for s in val:
                maskL = val < s
                maskR = val >= s
                NL = np.count_nonzero(maskL)
                NR = x - NL
                subsetL = subset[maskL, :]
                subsetR = subset[maskR, :]
                N1L = np.count_nonzero(subsetL[:, 3])
                N1R = np.count_nonzero(subsetR[:, 3])
                p1L = N1L/NL
                p1R = N1R/NR
                giniL = 1 - p1L**2 - (1 - p1L)**2
                giniR = 1 - p1R**2 - (1 - p1R)**2
                c = NL*giniL + NR*giniR
                if c < cmin:
                    kmin, smin, cmin = k, s, c
                    subL, subR = subsetL, subsetR
                    class1L, classe1R = N1L, N1R
                    nL, nR = NL, NR

        self.splittingFeature = kmin
        self.splittingValue = smin

        if self.depth == self.maxDepth - 1:
            self.left = 2*class1L >= nL
            self.right = 2*class1R >= nR

        if class1L == 0:
            self.left = False
        elif class1L == nL:
            self.left = True

        if classe1R == 0:
            self.right = False
        elif class1R == nR:
            self.right = True

        # Sinon enfants
        if self.left is None:
            self.left = BinaryTree(self.depth + 1)
            same, val = self.left.recSplit(subL)
            if same:
                self.left = val

        if self.right is None:
            self.right = BinaryTree(self.depth + 1)
            same, val = self.right.recSplit(subR)
            if same:
                self.right = val

        if (type(self.left) == bool and type(self.right) == bool and self.left == self.right):
            return True, self.left
        else:
            return False, None

    def apply(self, X):

        # Non vectorisé

        s = X.shape
        res = np.zeros(s[0])

        for k, x in enumerate(X):
            
            if x[self.splittingFeature] < self.splittingValue:
                if type(self.left) == bool:
                    res[k] = self.left
                else:
                    res[k] = self.left.apply(X)
            else:
                if type(self.left) == bool:
                    res[k] = self.right
                else:
                    res[k] = self.right.apply(X)

        return res
        

class RandomForest:

    def __init__(self):
        self.maxDepth = 6
        self.Ntrees = 200
        self.forest = []

    def __repr__(self) -> str:
        pass

    def __buildTree(self, trainSub):
        tree = BinaryTree(0, self.maxDepth)
        tree.recSplit(trainSub)
        return tree

    def __applyTree(self, X):
        cnt = 0
        for tree in self.forest:
            cnt += tree.apply(X)
        return cnt

    def train(self, trainSet):
        subList = trainSubset(trainSet, self.Ntrees)
        for trainSub in subList:
            self.forest.append(self.__buildTree(trainSub))

    def predict(self, X):
        return 2*self.__applyTree(X) > self.Ntrees

    def performance(self, testSet):
        sc = Scores(testSet[:, 3], self.predict(testSet[:, :3]))
        print(sc)
        return sc