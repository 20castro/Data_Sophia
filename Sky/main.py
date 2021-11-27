import numpy as np
import cProfile, pstats

from pickData import Collect
from algorithms.lda import LDA
from algorithms.random import RandomClassifier
from algorithms.qda import QDA
from algorithms.kernels import GaussianKernel
from algorithms.logit import Logit
from algorithms.forest import RandomForest
from algorithms.knn import KNN

def modelTest(model, train, test, model_name):
    profiler = cProfile.Profile()
    profiler.enable()
    model.train(train)
    #print(model)
    print(61*'_')
    print('\nScores for model ' + model_name)
    model.performance(test)
    profiler.disable()
    stats = pstats.Stats(profiler)
    #stats.dump_stats('file.bin')
    stats.print_stats(.2)
    #print(f'\nExecution time (prediction on test set): {1000*extime} ms\n')

def main(model_name):
    data = Collect()
    train, test = data.split(.05)
    if model_name == 'random':
        model = RandomClassifier()
    elif model_name == 'LDA':
        model = LDA()
    elif model_name == 'QDA':
        model = QDA()
    elif model_name == 'kernels':
        model = GaussianKernel() # arg is the expression of the kernel here (thus a function)
    elif model_name == 'logit':
        model = Logit()
    elif model_name == 'KNN':
        model = KNN()
    elif model_name == 'forest':
        model = RandomForest()
    else:
        raise NameError('Model not found')
    modelTest(model, train, test, model_name)

main('kernels')