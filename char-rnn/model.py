import tensorflow as tf
import tableprint as tp
import numpy as np
import time
from functools import partial
from yadll import lstm, affine
import data


class Model:
    def __init__(self, n_hidden=512, n_out=36):
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.sess = None
        self.loss_history = []

    def inference(self, xs):
        return lstm(xs, self.n_hidden, n_out=self.n_out)

    def loss(self, xs, ys):
        yhat = self.inference(xs)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=yhat)
        return tf.reduce_mean(loss)

    def sample(self, xseed, sample_length, fmap, temp=1.):
        bz, slen, nin = xseed.shape

        x_ph = tf.placeholder(tf.float32, shape=(bz, nin))
        state_a = tf.placeholder(tf.float32, shape=(bz, self.n_hidden))
        state_b = tf.placeholder(tf.float32, shape=(bz, self.n_hidden))
        state_ph = (state_a, state_b)
        state = (np.zeros((bz, n_units)), np.zeros((bz, n_units)))

        with tf.variable_scope("lstm"):

            # logits tensor
            with tf.variable_scope("rnn"):
                rnn = tf.contrib.rnn.BasicLSTMCell(n_units, reuse=True)
                y, new_state = rnn(x_ph, state_ph)
            output = affine(y, nin, scope="output_proj", reuse=True)

        # seed the state
        for idx in range(slen):
            feed = {state_a: state[0], state_b: state[1], x_ph: X[:, idx, :]}
            logits, state = self.sess.run([output, new_state], feed_dict=feed)

        # sample
        samples = [draw_sample(softmax(logits / temp))]
        for k in range(sample_length):
            x = np.zeros((bz, nin))
            for ix, s in enumerate(samples[-1]):
                x[ix, s] = 1.

            feed = {state_a: state[0], state_b: state[1], x_ph: x}
            logits, state = self.sess.run([output, new_state], feed_dict=feed)

            samples.append(draw_sample(softmax(logits / temp)))

        values = np.stack(samples).T
        return [''.join(map(fmap, v)) for v in values]

    def train(self, n_iter, datagen, sequence_length, batch_size):

        xs = tf.placeholder(tf.float32, shape=(batch_size, sequence_length, self.n_out))
        ys = tf.placeholder(tf.float32, shape=(batch_size, sequence_length, self.n_out))

        fobj = self.loss(xs, ys)
        train_op = tf.train.AdamOptimizer(2e-3).minimize(fobj)

        if self.sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

        losses = np.zeros(n_iter)
        width = (9, 11, 15)
        print(tp.header(['iter', 'loss', 'runtime'], width=width), flush=True)
        for k in range(n_iter):
            # Xtra, ytra = data.datagen(X, batch_size, seq_length)
            Xtra, ytra = datagen(batch_size, sequence_length)
            tstart = time.perf_counter()
            fk, _ = self.sess.run([fobj, train_op], feed_dict={xs: Xtra, ys: ytra})
            tstop = time.perf_counter() - tstart
            losses[k] = fk
            print(tp.row([k, fk, tp.humantime(tstop)], width=width), flush=True)
        print(tp.bottom(3, width=width))
        self.loss_history.append(losses)


def build(batch_size, sequence_length):
    N = 36

    xs = tf.placeholder(tf.float32, shape=(batch_size, sequence_length, N))
    ys = tf.placeholder(tf.float32, shape=(batch_size, sequence_length, N))

    n_hidden = 512
    yhat = lstm(xs, n_hidden, nout=N)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=yhat)
    fobj = tf.reduce_mean(loss)

    train_op = tf.train.AdamOptimizer(2e-3).minimize(fobj)

    return xs, ys, yhat, fobj, train_op


def softmax(logits):
    # subtract the max for numerical stability
    logits -= logits.max(axis=1).reshape(-1, 1)
    prob = np.exp(logits)
    prob /= prob.sum(axis=1).reshape(-1, 1)
    return prob


def draw_sample(prob):
    samples = []
    for p in prob:
        p /= p.sum()
        p -= 1e-6
        samples.append(np.argmax(np.random.multinomial(1, p)))
    return np.array(samples)


def main():
    nepochs = 1000
    niter = 100
    batch_size = 1000
    sequence_length = 50
    sample_length = 100
    nsamples = 100

    X, rev_lu = data.main()
    datagen = partial(data.datagen, X=X)
    xseed = datagen(nsamples, sequence_length)

    char_rnn = Model()

    for epoch in range(nepochs):
        print(tp.banner(f'Epoch #{epoch+1} of {nepochs}'))
        char_rnn.train(niter, datagen, sequence_length, batch_size)
        samples = char_rnn.sample(xseed, sample_length, rev_lu.__getitem__)
        txt = '\n--------------------\n'.join(samples)
        with open(f'epoch{epoch}.txt', 'w') as f:
            f.write(txt)

    return char_rnn


if __name__ == '__main__':
    loss_history = main()
