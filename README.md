### deep-learning-demos
demos of neural networks, in python

#### walkthrough
The `toynn.py` script is the main script, it generates data from a toy classification problem and trains a multilayered neural network to solve it.

The network is built using the code in `layers.py`, which contains two classes: a `Network` class which contains a stack of `Layer`s. Each layer implements one of the layers in the network, and keeps track of the parameters of that layer. the `Network` connects these layers together so that you can compute forward or backward passes through the network.

The `Layer` class lets you specify the size (# of input and output dimensions) for each layer. The nonlinearity is assumed to be a sigmoid (see the function in `utils.py`) and the parameters are optimized using [adam](http://arxiv.org/abs/1412.6980).

The toy classification problem consists of trying to classify points where the decision boundary is given by the 2D boundary of a norm ball.

#### setup
```bash
$ git clone https://github.com/nirum/deep-learning-demos
$ cd deep-learning-demos
$ pip install -r requirements.txt
```

#### requirements

- numpy
- matplotlib
- [tqdm](https://github.com/tqdm/tqdm)
- [toolz](https://github.com/pytoolz/toolz)
- [descent](https://github.com/nirum/descent)
