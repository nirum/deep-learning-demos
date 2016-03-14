import numpy as np
from descent.algorithms import adam
from toolz import compose
from utils import sigmoid

__all__ = ['Network', 'Layer']


class Network:

    def __init__(self, x, y, layers):
        """Builds a fully connected multilayer neural network"""

        self.x = x
        self.y = y
        self.layers = layers

    def loss(self, yhat):
        """Loss function"""

        # squared error
        # err = yhat - self.y
        # obj = 0.5 * np.mean(err ** 2)

        # cross-entropy
        obj = np.mean(-self.y * np.log(yhat) - (1-self.y) * np.log(1-yhat))
        err = (yhat - self.y) / (yhat * (1-yhat))

        return obj, err

    def predict(self, x=None):
        """Forward pass"""

        if x is None:
            x = self.x

        return compose(*reversed(self.layers))(x)

    def __call__(self):
        """Backward pass, computes error and runs backprop"""

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
    def __init__(self, dims, learning_rate=1e-3, activation=sigmoid):
        """A single fully connected layer"""

        # initialize parameters
        weight_scaling = 0.1 / np.sqrt(np.prod(dims))
        bias_scaling = 0.1 / np.sqrt(dims[0])
        self.weights =  np.random.randn(*dims) * weight_scaling
        self.bias = np.random.randn(dims[0],1) * bias_scaling

        # initialize nonlinearity
        self.activation = activation

        # initialize optimizer for each parameter
        self.weights_opt = adam(self.weights, lr=learning_rate)
        self.bias_opt = adam(self.bias, lr=learning_rate)

        # if true, then learning is turned on, and the weights are updated
        self.active = True

    def __call__(self, x):
        """Forward pass"""

        # store input
        self.x = x

        # affine term
        self.z = self.weights @ x + self.bias

        # apply activation nonlinearity and store derivative
        self.y, self.dy = self.activation(self.z)

        return self.y

    @property
    def shape(self):
        return self.weights.shape

    def backprop(self, error):
        """Backward pass"""

        nsamples = float(error.shape[1])
        delta = error * self.dy

        # parameter updates
        if self.active:
            dW = np.tensordot(delta, self.x, ([1], [1])) / nsamples
            db = np.sum(delta, axis=1).reshape(-1, 1) / nsamples
            self.update(dW, db)

        # pass back the new error
        return self.weights.T @ delta

    def update(self, dW, db):
        self.weights = self.weights_opt(dW)
        self.bias = self.bias_opt(db)
