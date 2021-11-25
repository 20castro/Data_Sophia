import numpy as np
import os
import matplotlib.pyplot as plt

from pickData import Collect
from algorithms.lda import LDA
from algorithms.random import RandomClassifier
from algorithms.qda import QDA
from algorithms.kernels import Kernels
from algorithms.logit import Logit

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
    elif model_name == 'logit':
        model = Logit()
    else:
        raise NameError('Model not found')

    model.train(train)

    # Choix d'une image et calcul du skymask

    dirs = [path for path in os.listdir('./Data') if path.startswith('ima')]
    src = np.random.choice(dirs)
    img = plt.imread('./Data/' + src)
    s = img.shape
    col = img.reshape((s[0]*s[1], 3))
    skymask = model.predict(col).reshape((s[0], s[1]))

    # Représentation

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(img)
    ax1.set_title('Image couleur')
    ax1.set_axis_off()

    ref = plt.imread('./Data/mask' + src[3:-4] + '_skymask.png')
    ax2.imshow(ref, cmap='binary')
    ax2.set_title('Skymask obtenu à la main')
    ax2.set_axis_off()

    ax3.imshow(skymask, cmap='binary')
    ax3.set_title('Skymask calculé avec le model ' + model_name)
    ax3.set_axis_off()

    fig.tight_layout()
    plt.show()

build('QDA')