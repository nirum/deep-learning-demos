"""
Toy fully-connected multi-layered neural networks

"""

import numpy as np
import matplotlib.pyplot as plt
from utils import generate_data
from layers import Layer, Network
from tqdm import trange


def build_fullyconnected(norm=np.inf, nhidden=5, fa1=True, fa2=True):

    # generate data (in a box)
    X, y = generate_data(norm=norm)

    # build network
    L1 = Layer((nhidden, 2), feedback_alignment=fa1)
    L2 = Layer((1, nhidden), feedback_alignment=fa2)
    net = Network(X, y, [L1, L2])

    return net


def sim(norm, nh, fa1, fa2, numiter=10000):
    net = build_fullyconnected(norm=norm, nhidden=nh, fa1=fa1, fa2=fa2)
    return np.array([net() for _ in trange(numiter)]), net


if __name__ == '__main__':

    # norm ball for generate toy data
    norm = np.inf

    objective, net = sim(norm, 50, True, True)

    # predicted class labels (on held out data)
    X_holdout, y_holdout = generate_data(norm=norm, nsamples=5000)
    yhat = net.predict(X_holdout)[0]

    # plot the training curve
    plt.figure()
    plt.plot(np.arange(objective.size), objective)
    plt.xlabel('Iteration ($k$)')
    plt.ylabel('Training error ($f(k)$)')

    # plot labeled training data
    plt.figure()
    plt.scatter(X_holdout[0], X_holdout[1], s=50, c=yhat, cmap='seismic')
    plt.gca().set_aspect('equal')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    plt.show()
    plt.draw()
