import tensorflow as tf
import tableprint as tp
import numpy as np
import time
from yadll import lstm, affine
import data


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


def train(niter, xs, ys, X, fobj, train_op, sess=None):

    batch_size, seq_length, _ = xs.shape.as_list()

    if sess is None:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

    losses = np.zeros(niter)
    width = (9, 11, 15)
    print(tp.header(['iter', 'loss', 'runtime'], width=width), flush=True)
    for k in range(niter):
        Xtra, ytra = data.datagen(X, batch_size, seq_length)
        tstart = time.perf_counter()
        fk, _ = sess.run([fobj, train_op], feed_dict={xs: Xtra, ys: ytra})
        tstop = time.perf_counter() - tstart
        losses[k] = fk
        print(tp.row([k, fk, tp.humantime(tstop)], width=width), flush=True)
    print(tp.bottom(3, width=width))
    return sess, losses


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


def sample(sess, nsamples, X, rev_lu, temp=1.):

    bz, slen, nin = X.shape
    n_units = 512

    x_ph = tf.placeholder(tf.float32, shape=(bz, nin))
    state_a = tf.placeholder(tf.float32, shape=(bz, n_units))
    state_b = tf.placeholder(tf.float32, shape=(bz, n_units))
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
        logits, state = sess.run([output, new_state], feed_dict=feed)

    # sample
    samples = [draw_sample(softmax(logits / temp))]
    for k in range(nsamples):
        x = np.zeros((bz, nin))
        for ix, s in enumerate(samples[-1]):
            x[ix, s] = 1.

        feed = {state_a: state[0], state_b: state[1], x_ph: x}
        logits, state = sess.run([output, new_state], feed_dict=feed)

        samples.append(draw_sample(softmax(logits / temp)))

    values = np.stack(samples).T
    return [''.join(map(rev_lu.__getitem__, v)) for v in values]


def main():
    nepochs = 1000
    niter = 100
    batch_size = 1000
    sequence_length = 50
    sample_length = 100
    nsamples = 1000

    sess = None
    X, rev_lu = data.main()
    xs, ys, yhat, fobj, train_op = build(batch_size, sequence_length)
    xseed, _ = data.datagen(X, nsamples, sequence_length)

    loss_history = []
    for epoch in range(nepochs):
        sess, losses = train(niter, xs, ys, X, fobj, train_op, sess=sess)
        samples = sample(sess, sample_length, xseed, rev_lu, temp=0.01)
        txt = '\n--------------------\n'.join(samples)
        loss_history.append(losses)
        with open(f'epoch{epoch}.txt', 'w') as f:
            f.write(txt)

    return loss_history


if __name__ == '__main__':
    loss_history = main()
