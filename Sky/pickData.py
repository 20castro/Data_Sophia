import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

## Création du dataset

# Récupération des images

def collectData() -> np.ndarray:

    dirs = os.listdir('./Data')

    length = 0
    for src in dirs:
        if src.startswith('ima'):
            img = plt.imread('./Data/' + src)
            s = img.shape
            length += s[0]*s[1]

    data = np.zeros((length, 4))
    cur = 0
    for src in dirs:
        if src.startswith('ima'):

            img = plt.imread('./Data/' + src)
            s = img.shape
            add = s[0]*s[1]
            data[cur:cur + add, :3] = img.reshape((add, 3))/255

            mask = plt.imread('./Data/mask' + src[3:-4] + '_skymask.png')
            data[cur:cur + add, 3] = mask.reshape((1, add))

            cur += add

    np.save('./Data/pack.npy', data)
    return data

class Collect:

    def __init__(self):
        try:
            self.data = np.load('./Data/pack.npy')
        except FileNotFoundError:
            self.data = collectData()
        self.length = len(self.data)

    def __repr__(self):
        df = pd.DataFrame(data=self.data, columns=['R', 'G', 'B', 'label'])
        print(df.head(), end='\n\n')
        return f'{self.length} images found'

    def split(self, proportion):

        '''Jeux d'entraînement et de test'''

        assert 0 < proportion < 1

        N = int(self.length*proportion)
        index = np.random.choice(self.length, N)
        mask = np.zeros(self.length).astype('bool')
        mask[index] = True
        return self.data[mask], self.data[np.logical_not(mask)]