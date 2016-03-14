"""
Toy fully-connected multi-layered neural networks

"""

import numpy as np
import matplotlib.pyplot as plt
from utils import generate_data
from layers import Layer, Network
from tqdm import trange


def build_fullyconnected(norm=np.inf, nhidden=5):

    # generate data (in a box)
    X, y = generate_data(norm=norm)

    # build network
    L1 = Layer((nhidden, 2))
    L2 = Layer((1, nhidden))
    net = Network(X, y, [L1, L2])

    return net


if __name__ == '__main__':

    # build the network
    net = build_fullyconnected()

    # train
    numiter = 10000
    objective = np.array([net() for _ in trange(numiter)])

    # predicted class labels
    yhat = net.predict()[0]

    # plot the training curve
    plt.figure()
    plt.plot(np.arange(numiter), objective)
    plt.xlabel('Iteration ($k$)')
    plt.ylabel('Training error ($f(k)$)')

    # plot labeled training data
    plt.figure()
    plt.plot(net.x[0][yhat > 0.5], net.x[1][yhat > 0.5], 'ro')
    plt.plot(net.x[0][yhat <= 0.5], net.x[1][yhat <= 0.5], 'bo')
