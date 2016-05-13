import cPickle

import skimage.transform
import skimage.color
from PIL import Image

import theano
import theano.tensor as T
import theano.tensor.nlinalg as linalg
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

def unpickle(file, modifier):
    fo = open(file, modifier)
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_data(file, modifier):
    # load the data
    batch = unpickle(file, modifier)

    # Populate data arrays, where the rows represent the data features
    # and the columns represent the data samples
    train_set_x = batch['data']

    # Rescale the images and convert them into gray scale
    X_scaled = np.zeros((train_set_x.shape[0], 12*12))
    size = 12, 12
    for i in range(train_set_x.shape[0]):
        cur_img = np.reshape(train_set_x[i], (32, 32, 3), order='F')
        im = Image.fromarray(cur_img)
        im.thumbnail(size, Image.ANTIALIAS)
        X_scaled[i] = skimage.color.rgb2gray(skimage.transform.resize(cur_img, (12, 12))).flatten()

    return theano.shared(value=np.transpose(X_scaled), name='train_set_x', borrow=True)


def kmeans(train_set_x):

    if train_set_x is None:
        train_set_x = T.matrix('train_set_x')

    ########################
    # Normalize the inputs #
    ########################

    epsilon_norm = 10
    epsilon_zca = 0.015
    K = 500

    train_set_x = train_set_x - T.mean(train_set_x, axis=0) / T.sqrt(T.var(train_set_x, axis=0) + epsilon_norm)

    #####################
    # Whiten the inputs #
    #####################

    # a simple choice of whitening transform is the ZCA whitening transform
    # epsilon_zca is small constant
    # for contrast-normalizaed data, setting epsilon_zca to 0.01 for 16-by-16 pixel patches,
    #                                                 or to  0.1 for 8-by-8   pixel patches
    # is good starting point
    cov = T.dot(train_set_x, T.transpose(train_set_x)) / train_set_x.shape[1]
    U, S, V = linalg.svd(cov)
    tmp = T.dot(U, T.diag(1/T.sqrt(S + epsilon_zca)))
    tmp = T.dot(tmp, T.transpose(U))
    whitened_x = T.dot(tmp, train_set_x)

    ######################
    # Training the Model #
    ######################

    # Initialization
    dimension_size = whitened_x.shape[0]
    num_samples = whitened_x.shape[1]
    srng = RandomStreams(seed=234)

    D = srng.normal(size=(dimension_size, K))
    D = D / T.sqrt(T.sum(T.sqr(D), axis=0))

    # typically 10 iterations is enough
    num_iteration = 15

    # compute new centroids, D_new
    for i in xrange(num_iteration):

        dx = T.dot(D.T, whitened_x)
        arg_max_dx = T.argmax(dx, axis=0)
        s = dx[arg_max_dx, T.arange(num_samples)]

        S = T.zeros((K, num_samples))
        S = T.set_subtensor(S[arg_max_dx, T.arange(num_samples)], s)
        D = T.dot(whitened_x, T.transpose(S)) + D

        D = D / T.sqrt(T.sum(T.sqr(D), axis=0))

    return D

if __name__ == '__main__':

    train_set_x = load_data('cifar-10-batches-py/data_batch_1', 'rb')
    X = T.matrix('X', dtype='float64')

    result = theano.function(
        inputs = [],
        outputs = kmeans(train_set_x = train_set_x),
        givens = {
            X: train_set_x
        }
    )

    D = result()