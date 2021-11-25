import numpy as np

from pickData import Collect
from algorithms.lda import LDA
from algorithms.random import RandomClassifier
from algorithms.qda import QDA
from algorithms.kernels import Kernels

def gaussian(u):
    return np.exp(-.5*u.dot(u))/np.sqrt(2*np.pi)

def modelTest(model, train, test, model_name):
    model.train(train)
    # print(model, end='\n\n')
    print(61*'_')
    print('\nScores for model ' + model_name)
    extime = model.performance(test)
    print(f'\nExecution time (prediction on test set): {1000*extime} ms\n')

def main(model_name, arg=None):
    data = Collect()
    train, test = data.split(.1)
    if model_name == 'random':
        model = RandomClassifier()
    elif model_name == 'LDA':
        model = LDA()
    elif model_name == 'QDA':
        model = QDA()
    elif model_name == 'kernels':
        model = Kernels(arg) # arg is the expression of the kernel here (thus a function)
    else:
        raise NameError('Model not found')
    modelTest(model, train, test, model_name)

main('random')