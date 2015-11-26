import numpy as np
from descent.algorithms import adam
from toolz import compose
from utils import sigmoid

__all__ = ['Network', 'Layer', 'RecurrentLayer']


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
    def __init__(self, dims, learning_rate=1e-3, activation=sigmoid):

        # initialize parameters
        weight_scaling = 0.1 / np.sqrt(np.prod(dims))
        bias_scaling = 0.1 / np.sqrt(dims[0])
        self.weights =  np.random.randn(*dims) * weight_scaling
        self.bias = np.random.randn(dims[0],1) * bias_scaling

        # initialize nonlinearity
        self.activation = activation

        # initialize learning algorithm
        self.weights_opt = adam(lr=learning_rate)
        self.weights_opt.send(self.weights)

        self.bias_opt = adam(lr=learning_rate)
        self.bias_opt.send(self.bias)

        self.active = True

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
        if active:
            dW = np.tensordot(delta, self.x, ([1], [1])) / nsamples
            db = np.sum(delta, axis=1).reshape(-1, 1) / nsamples
            self.update(dW, db)

        # pass back the new error
        errornext = self.weights.T @ delta

        return errornext

    def update(self, dW, db):
        self.weights = self.weights_opt.send(dW)
        self.bias = self.bias_opt.send(db)


class RecurrentLayer:

    def __init__(self, seq_length, dims, learning_rate=1e-3, activation=sigmoid):

        # sequence length
        self.T = seq_length

        # input weights
        self.W =  np.random.randn(*dims) * 0.1 / np.sqrt(np.prod(dims))

        # recurrent weights
        self.U =  np.random.randn(dims[0], dims[0]) * 0.1 / float(dims[0])

        # bias
        self.b = np.random.randn(dims[0],) * 0.1 / np.sqrt(dims[0])

        # nonlinearity
        self.activation = activation

        # initialize learning algorithm
        self.W_opt = adam(lr=learning_rate)
        self.W_opt.send(self.W)

        self.U_opt = adam(lr=learning_rate)
        self.U_opt.send(self.U)

        self.b_opt = adam(lr=learning_rate)
        self.b_opt.send(self.b)

        self.active = True

    def __call__(self, xs):
        """
        Forward pass
        """

        # store input
        self.xs = xs

        # store affine term at each step in the sequence
        self.z = []
        self.y = []
        self.dy = []

        # feed in the sequence
        for t, x in enumerate(xs):

            if t == 0:
                z = self.W @ x + self.b

            else:
                z = self.W @ x + self.U @ self.y[t-1] + self.b

            y, dy = self.activation(z)

            self.z.append(z)
            self.y.append(y)
            self.dy.append(dy)

        # return final output
        return self.y[-1]

    def backprop(self, error):
        """
        Backward pass
        """

        dW = np.zeros_like(self.W)
        dU = np.zeros_like(self.U)
        db = np.zeros_like(self.b)

        for t in reversed(range(self.T)):

            delta = error * self.dy[t]
            db += delta
            dW += np.outer(delta, self.xs[t])
            if t > 0:
                dU += np.outer(delta, self.y[t-1])

            error = self.U.T @ delta

        # normalize
        dW /= float(self.T)
        dU /= float(self.T-1)
        db /= float(self.T)

        # update the parameters
        if self.active:
            self.update(dW, dU, db)

            # pass back the new error
            return self.W.T @ delta

        else:

            return dW, dU, db

    def update(self, dW, dU, db):
        self.W = self.W_opt.send(dW)
        self.U = self.U_opt.send(dU)
        self.b = self.b_opt.send(db)

