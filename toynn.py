"""
Toy fully-connected multi-layered neural networks

"""

import numpy as np
import matplotlib.pyplot as plt
from jetpack.chart import breathe, noticks
from utils import generate_data
from layers import *


def build_fullyconnected(norm=np.inf, nhidden=5):

    # generate data (in a box)
    X, y = generate_data(norm=norm)

    # build network
    L1 = Layer((nhidden,2))
    L2 = Layer((1,nhidden))
    net = Network(X, y, [L1, L2])

    return net


def build_recurrent():

    L1 


if __name__ == '__main__':

    net = build_fullyconnected()
