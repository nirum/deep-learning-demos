import numpy as np


def generate_data(norm=np.inf, nsamples=1000):
    """Generates 2D labeled data, where the label depends on the location of
    the point relative to a norm ball"""

    # generate 2D data (decision boundary is given by a norm ball)
    X = np.random.rand(2, nsamples) * 2 - 1
    y = np.ones(X.shape[1]) * (np.linalg.norm(X, norm, axis=0) > 0.75)

    return X, y


def sigmoid(u):
    """Sigmoidal nonlinearity, with gradient"""
    g = 1 / (1 + np.exp(-u))
    return g, g * (1 - g)
