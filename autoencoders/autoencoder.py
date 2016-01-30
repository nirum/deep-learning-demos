from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, AutoEncoder
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import containers
from keras.optimizers import adam
from keras.datasets import mnist

def setup(ndim):

    forward = [
        Dense(512, input_dim=ndim),
        Dense(256),
    ]

    backward = [
        Dense(512, input_dim=256),
        Dense(ndim),
    ]

    return build(forward, backward)


def build(forward, backward):
    """
    Builds an autoencoder
    """

    print('Building...', end='', flush=True)

    encoder = containers.Sequential(forward)
    decoder = containers.Sequential(backward)

    model = Sequential()
    model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))

    model.compile(loss='mean_squared_error', optimizer=adam())

    print('Done.', flush=True)

    return model


def run_mnist():
    """
    Trains an autoencoder on MNIST
    """

    # reshape the data
    (X_train, _), (X_test, _) = mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype("float32")
    X_test = X_test.reshape(-1, 784).astype("float32")

    # build the model
    ae = setup(784)

    # train
    ae.fit(X_train, X_train, batch_size=1000, nb_epoch=100, show_accuracy=False, verbose=1, validation_data=[X_test, X_test])

    return ae


if __name__ == "__init__":
    ae = run_mnist()
