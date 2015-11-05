import numpy as np

def generate_data(norm=np.inf, nsamples=1000):

    # generate data (in a box)
    X = np.random.rand(2, nsamples)*2 - 1
    y = np.ones(X.shape[1]) * (np.linalg.norm(X, norm, axis=0) > 0.75)

    return X, y



