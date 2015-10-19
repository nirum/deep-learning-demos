"""
Toy fully-connected multi-layered neural networks

"""

import numpy as np
from descent import Adam


class Network:

    def __init__(self, dim):

        self.layers = []

        for ix in range(len(dims) - 1):
            self.layers.append(Layer(dims[ix:(ix+2)]))


class Layer:
    def __init__(self, dims, learning_rate=1e-3):

        self.nin = dims[1]
        self.nout = dims[0]

        weight_scaling = 0.1 / np.sqrt(np.prod(dims))
        bias_scaling = 0.1 / np.sqrt(dims[0])

        self.weights =  np.random.randn(*dims) * weight_scaling
        self.bias = np.random.randn(dims[0],1) * bias_scaling

    def __call__(self, x):

        # store input
        self.x = x

        # affine term
        self.z = self.weights @ x + self.bias

        # nonlinearity and gradient
        self.y, self.dy = self.activation(self.z)

        return self.y

    def activation(self, u):
        g = 1 / (1 + np.exp(-u))
        return g, g*(1-g)

    @property
    def shape(self):
        return self.weights.shape

    def backprop(self, error):

        nsamples = float(error.shape[1])
        tmp = error * self.dy
        dW = np.tensordot(tmp, self.x, ([1], [1])) / nsamples
        db = np.sum(tmp, axis=1).reshape(-1,1) / nsamples
        delta = self.weights.T @ error

        return dW, db, delta

    def update(self, weights, bias):
        self.weights = weights
        self.bias = bias


if __name__ == '__main__':

    # L1 = layer((1,2))

    # generate data (in a box)
    X = np.random.rand(2, 1000)*2 - 1
    y = np.ones(X.shape[1]) * (np.linalg.norm(X, axis=0) > 0.75)

    # def net_obj(W):
        # L1.W = W
        # yhat = L1(X)
        # err = y - yhat
        # obj = 0.5 * np.mean(err ** 2)
        # grad = L1.backprop(err)
        # return obj, grad

    # optimize
    # opt = Adam(net_obj, np.random.randn(*L1.shape), learning_rate=0.01)
    # opt.run()
