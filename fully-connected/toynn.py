"""
Toy fully-connected multi-layered neural networks

"""

import numpy as np
import matplotlib.pyplot as plt
from descent import GradientDescent
from descent.algorithms import adam
from toolz import compose
from jetpack.chart import breathe, noticks
from utils import generate_data


class Network:

    def __init__(self, x, y, layers):

        self.x = x
        self.y = y
        self.layers = layers

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

    def predict(self, x=None):

        if x is None:
            x = self.x

        return compose(*reversed(self.layers))(x)

    def __call__(self):
        """
        Objective and gradient

        """

        # compute prediction from forward pass
        yhat = compose(*reversed(self.layers))(self.x)

        # evaluate the prediction on the loss function
        obj, error = self.loss(yhat)

        # backward pass
        for layer in reversed(self.layers):

            # backpropogate the error
            error = layer.backprop(error)

        return obj


class Layer:
    def __init__(self, dims, learning_rate=1e-3):

        # initialize parameters
        weight_scaling = 0.1 / np.sqrt(np.prod(dims))
        bias_scaling = 0.1 / np.sqrt(dims[0])
        self.weights =  np.random.randn(*dims) * weight_scaling
        self.bias = np.random.randn(dims[0],1) * bias_scaling

        # initialize learning algorithm
        self.weights_opt = adam(lr=learning_rate)
        self.weights_opt.send(self.weights)

        self.bias_opt = adam(lr=learning_rate)
        self.bias_opt.send(self.bias)

    def __call__(self, x):
        """
        Forward pass
        """

        # store input
        self.x = x

        # affine term
        self.z = self.weights @ x + self.bias

        # apply activation nonlinearity and store derivative
        self.y, self.dy = self.activation(self.z)

        return self.y

    def activation(self, u):
        """
        Sigmoidal nonlinearity
        """
        g = 1 / (1 + np.exp(-u))
        return g, g*(1-g)

    @property
    def shape(self):
        return self.weights.shape

    def backprop(self, error):
        """
        Backward pass
        """

        nsamples = float(error.shape[1])
        delta = error * self.dy

        # parameter updates
        dW = np.tensordot(delta, self.x, ([1], [1])) / nsamples
        db = np.sum(delta, axis=1).reshape(-1, 1) / nsamples
        self.update(dW, db)

        # pass back the new error
        errornext = self.weights.T @ delta

        return errornext

    def update(self, dW, db):
        self.weights = self.weights_opt.send(dW)
        self.bias = self.bias_opt.send(db)


if __name__ == '__main__':

    # generate data (in a box)
    X, y = generate_data(norm=np.inf)

    # build network
    L1 = Layer((5,2))
    L2 = Layer((1,5))
    net = Network(X, y, [L1, L2])

    # optimize
    # opt = GradientDescent(net.theta_init, net, adam(lr=1e-2))

    # contour
    # V = [0., 0.25, 0.5, 0.75, 1.]
    # xm, ym = np.meshgrid(np.linspace(-1,1,200), np.linspace(-1,1,200))
    # Xm = np.stack([xm.ravel(), ym.ravel()])

    # def callback(d, every=100):
        # if d.iteration % every == 0:
            # ygrid = net.predict(d.params, Xm)
            # plt.contourf(xm, ym, ygrid.reshape(200,200), V, cmap='RdBu')
            # plt.gca().set_aspect('equal')
            # noticks()
            # plt.savefig('figures/iter{:05d}.png'.format(d.iteration), bbox_inches='tight', dpi=150)
            # plt.close()

    # # append callback
    # opt.callbacks.append(callback)
