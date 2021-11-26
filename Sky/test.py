import numpy as np
from time import perf_counter

from scores import Scores
from pickData import Collect
from algorithms.logit import log_likelihood_derivatives

def test_scores():
    lab = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 1, 0])
    s = Scores(lab, pred)
    print(s)

def test_collect():
    data = Collect()
    print(data)

def broadcasting():
    data = Collect()
    trainSet, _ = data.split(.02)
    s = trainSet.shape
    u = np.ones(s)
    u[:, 1:] = trainSet[:, :3] # première colonne de 1, le reste rempli par les pixels
    mask1 = trainSet[:, 3] == 1
    mask0 = trainSet[:, 3] == 0
    start = perf_counter()
    d1, d2 = log_likelihood_derivatives(np.array([0, 0, 0, 0]), u[mask1], u[mask0])
    end = perf_counter()
    print(d1)
    print(d2)
    print(f'Time for one calculus : {1000*(end - start)} ms')

def run_tests():
    # test_scores()
    # test_collect()
    broadcasting()

run_tests()