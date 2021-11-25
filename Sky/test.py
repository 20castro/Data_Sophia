import numpy as np

from scores import Scores
from pickData import Collect

def test_scores():
    lab = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 1, 0])
    s = Scores(lab, pred)
    print(s)

def test_collect():
    data = Collect()
    print(data)

def run_tests():
    test_scores()
    test_collect()