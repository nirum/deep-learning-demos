"""
multi-layered neural network in keras

"""

import numpy as np
import matplotlib.pyplot as plt
from descent import Adam
from toolz import compose
from jetpack.chart import breathe, noticks

from keras.models import Sequential
from keras.layers.core import Dense, Flatten

from utils import generate_data


def mlp(nhidden=5):
    mdl = Sequential()
    mdl.add(Dense(nhidden, input_shape=(2,), activation='tanh'))
    mdl.add(Dense(1, activation='tanh'))
    mdl.compile(loss='binary_crossentropy', optimizer='adam')
    return mdl

if __name__ == '__main__':

    mdl = mlp(nhidden=25)
    X, y = generate_data()

    every = 100

    V = [0., 0.25, 0.5, 0.75, 1.]
    xm, ym = np.meshgrid(np.linspace(-1,1,200), np.linspace(-1,1,200))
    Xm = np.stack([xm.ravel(), ym.ravel()])

    loss = np.zeros(1e5)
    for j in range(loss.size):

        loss[j] = mdl.train_on_batch(X.T, y)

        if j % every == 0:
            yhat = mdl.predict(Xm.T)
            plt.contourf(xm, ym, yhat.reshape(200,200), V, cmap='RdBu')
            plt.gca().set_aspect('equal')
            noticks()
            plt.savefig('figures/iter{:05d}.png'.format(j), bbox_inches='tight', dpi=150)
            plt.close()
