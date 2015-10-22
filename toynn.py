"""
Toy fully-connected multi-layered neural networks

"""

import numpy as np
import matplotlib.pyplot as plt
from descent import Adam
from toolz import compose
from jetpack.chart import breathe, noticks


class Network:

    def __init__(self, x, y, layers):

        self.x = x
        self.y = y
        self.layers = layers

        # initial parameters
        self.theta_init = [{'weights': layer.weights, 'bias': layer.bias}
                           for layer in layers]

    def loss(self, yhat):
        """
        Loss function

        """

        # squared error
        # err = yhat - self.y
        # obj = 0.5 * np.mean(err ** 2)

        # cross-entropy
        obj = np.mean(-self.y * np.log(yhat) - (1-self.y) * np.log(1-yhat))
        err = (yhat - self.y) / (yhat * (1-yhat))

        return obj, err

    def predict(self, theta, x=None):

        # update parameters in each layer
        for layer, theta in zip(self.layers, theta):
            layer.update(theta['weights'], theta['bias'])

        if x is None:
            x = self.x

        return compose(*reversed(self.layers))(x)

    def __call__(self, theta):
        """
        Objective and gradient

        """

        # update parameters in each layer
        for layer, theta in zip(self.layers, theta):
            layer.update(theta['weights'], theta['bias'])

        # compute prediction from forward pass
        yhat = compose(*reversed(self.layers))(self.x)

        # evaluate the prediction on the loss function
        obj, error = self.loss(yhat)

        # backward pass
        gradient = []
        for layer in reversed(self.layers):

            # backpropogate the error
            dW, db, error = layer.backprop(error)

            # append to the gradient
            gradient.append({'weights': dW, 'bias': db})

        return obj, list(reversed(gradient))


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
        delta = self.weights.T @ tmp

        return dW, db, delta

    def update(self, weights, bias):
        self.weights = weights.copy()
        self.bias = bias.copy()


if __name__ == '__main__':

    # generate data (in a box)
    X = np.random.rand(2, 1000)*2 - 1
    y = np.ones(X.shape[1]) * (np.linalg.norm(X, np.inf, axis=0) > 0.75)

    # build network
    L1 = Layer((5,2))
    L2 = Layer((1,5))
    net = Network(X, y, [L1, L2])

    # optimize
    opt = Adam(net, net.theta_init, learning_rate=1e-3)
    opt.display.every = 100

    # contour
    V = [0., 0.25, 0.5, 0.75, 1.]
    xm, ym = np.meshgrid(np.linspace(-1,1,200), np.linspace(-1,1,200))
    Xm = np.stack([xm.ravel(), ym.ravel()])

    def callback(d, every=100):
        if d.iteration % every == 0:
            ygrid = net.predict(d.params, Xm)
            plt.contourf(xm, ym, ygrid.reshape(200,200), V, cmap='RdBu')
            plt.gca().set_aspect('equal')
            noticks()
            plt.savefig('figures/iter{:05d}.png'.format(d.iteration), bbox_inches='tight', dpi=150)
            plt.close()

    # append callback
    opt.callbacks.append(callback)
