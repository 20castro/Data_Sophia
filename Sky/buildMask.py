import numpy as np
import os
import matplotlib.pyplot as plt

from pick_data import Collect
from algorithms.lda import LDA
from algorithms.random import RandomClassifier
from algorithms.qda import QDA
from algorithms.kernels import Kernels

def build(model_name, arg=None):

    # Entraînement

    data = Collect()
    train, _ = data.split(.1)
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

    model.train(train)

    # Choix d'une image

    dirs = [path for path in os.listdir('./Data') if path.startswith('ima')]
    src = np.random.choice(dirs)
    img = plt.imread('./Data/' + src)
    s = img.shape
    col = img.reshape((s[0]*s[1], 3))
    skymask = model.predict(col).reshape(s)

    plt.imshow(img)
    ref = plt.imread('./Data/mask' + src[3:-4] + '_skymask.png')
    plt.imshow(ref, cmap='binary')
    plt.imshow(skymask, cmap='binary')