import os
import numpy as np
from collections import Counter


def loadtxt(filename):
    with open(filename, 'r') as f:
        string = sanitize(''.join(f.readlines()))
    return string


def sanitize(string):
    stripchars = ['3', '&', '$']
    string = str.lower(string)
    for ch in stripchars:
        string = string.replace(ch, '')
    return string


def datagen(X, batch_size, sequence_length):
    start_inds = np.random.randint(X.shape[0] - sequence_length, size=batch_size)
    xs = [X[i:(i + sequence_length), :] for i in start_inds]
    ys = [X[(i + 1):(i + 1 + sequence_length), :] for i in start_inds]
    return np.stack(xs), np.stack(ys)


def main():

    # load txt
    ss = loadtxt(os.path.expanduser('~/data/shakespeare.txt'))
    cc = Counter(ss)
    N = len(cc.keys())

    # embedding
    lookup = dict(zip(cc.keys(), np.eye(N)))
    rev_lu = {v.argmax(): k for k, v in lookup.items()}
    X = np.stack([lookup[ch] for ch in ss])

    return X, rev_lu


if __name__ == '__main__':
    X, rev_lu = main()
