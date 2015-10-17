"""
Toy neural networks
"""

import numpy as np
from descent import Adam


class Network:

    def __init__(self, dim):

        self.layers = []

        for ix in range(len(dims) - 1):
            self.layers.append(Layer(dims[ix:(ix+2)]))


class Layer:
    def __init__(self, dims):
        self.W = (0.1 / np.sqrt(np.prod(dims))) * np.random.randn(*dims)

    def __call__(self, x):
        self.z = self.W @ x
        self.y, self.dy = self.f_df(self.z); self.x = x
        return self.y

    def f_df(self, u):
        obj = 1 / (1 + np.exp(-u)); return obj, obj*(1-obj)

    @property
    def shape(self):
        return self.W.shape

    def backprop(self, err):

        # delta to pass back
        self.delta = (self.W.T @ err) * self.dy

        # weight update
        self.dW = -((err * self.dy) @ self.x.T) / float(err.size)
        return self.dW


if __name__ == '__main__':

    L1 = layer((1,2))

    # generate data (in a box)
    X = np.random.rand(2, 1000)*2 - 1
    y = np.ones(X.shape[1]) * (np.linalg.norm(X, axis=0) > 0.75)

    def net_obj(W):
        L1.W = W
        yhat = L1(X)
        err = y - yhat
        obj = 0.5 * np.mean(err ** 2)
        grad = L1.backprop(err)
        return obj, grad

    # optimize
    opt = Adam(net_obj, np.random.randn(*L1.shape), learning_rate=0.01)
    opt.run()
