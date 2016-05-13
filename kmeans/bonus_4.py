import cPickle
from utils import tile_raster_images

# Theano and numpy
import theano
import theano.tensor as T
import theano.tensor.nlinalg as linalg
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

# Image processing
import skimage.transform
import skimage.color
from PIL import Image

def unpickle(file, modifier):
    fo = open(file, modifier)
    dict = cPickle.load(fo)
    fo.close()
    return dict


def load_data(file, modifier):
    # load the data
    batch = unpickle(file, modifier)

    train_set_x = batch['data']

    X_scaled = np.zeros((train_set_x.shape[0], 12*12))
    size = 12, 12
    for i in range(train_set_x.shape[0]):
        cur_img = np.reshape(train_set_x[i], (32, 32, 3), order='F')
        im = Image.fromarray(cur_img)
        im.thumbnail(size, Image.ANTIALIAS)
        X_scaled[i] = skimage.color.rgb2gray(skimage.transform.resize(cur_img, (12, 12))).flatten()

    # return theano.shared(value=np.transpose(X_scaled), name='train_set_x', borrow=True)
    return X_scaled

def get_batch(data):
    return np.transpose(data[np.random.choice(10000, 1000, replace=False)])

class kmean_mini_batch(object):

    def __init__(self, batch_size, data=None):

        if data is None:
            self.train_set_x = T.matrix('train_set_x')
        else:
            self.train_set_x = data

        ########################
        # Normalize the inputs #
        ########################

        self.epsilon_norm = 10
        self.epsilon_zca = 0.015
        self.K = 500

        self.train_set_x = self.train_set_x - T.mean(self.train_set_x, axis=0) / T.sqrt(T.var(self.train_set_x, axis=0) + self.epsilon_norm)

        #####################
        # Whiten the inputs #
        #####################

        # a simple choice of whitening transform is the ZCA whitening transform
        # epsilon_zca is small constant
        # for contrast-normalizaed data, setting epsilon_zca to 0.01 for 16-by-16 pixel patches,
        #                                                 or to  0.1 for 8-by-8   pixel patches
        # is good starting point
        cov = T.dot(self.train_set_x, T.transpose(self.train_set_x)) / self.train_set_x.shape[1]
        U, S, V = linalg.svd(cov)
        tmp = T.dot(U, T.diag(1 / T.sqrt(S + self.epsilon_zca)))
        tmp = T.dot(tmp, T.transpose(U))
        self.whitened_x = T.dot(tmp, self.train_set_x)

        ######################
        # Training the Model #
        ######################

        # initialization
        self.dimension_size = self.train_set_x.shape[0]
        self.num_samples = batch_size
        self.srng = RandomStreams(seed=234)

        # We initialize the centroids by sampling them from a normal
        # distribution, and then normalizing them to unit length
        # D \in R^{n \times k}
        self.D = self.srng.normal(size=(self.dimension_size, self.K))
        self.D = self.D / T.sqrt(T.sum(T.sqr(self.D), axis=0))

    def training_batch(self):

        # Initialize new point representations
        # for every pass of the algorithm


        dx = T.dot(self.D.T, self.whitened_x)
        arg_max_dx = T.argmax(dx, axis=0)
        s = dx[arg_max_dx, T.arange(self.num_samples)]

        S = T.zeros((self.K, self.num_samples))
        S = T.set_subtensor(S[arg_max_dx, T.arange(self.num_samples)], s)
        self.D = T.dot(self.whitened_x, T.transpose(S)) + self.D

        self.D = self.D / T.sqrt(T.sum(T.sqr(self.D), axis=0))

        return self.D

if __name__ == '__main__':
    X = T.matrix('X', dtype='float64')
    mini_batch = T.matrix('mini_batch', dtype='float64')

    m_batch_size = 1000
    m_num_iter = 50000/m_batch_size


    kmeans = kmean_mini_batch(
        batch_size=m_batch_size,
        data=X,
    )

    func = theano.function(
        inputs=[mini_batch],
        outputs=kmeans.training_batch(),
        givens={
            X: mini_batch
        },
    )

    train_set_x = load_data('cifar-10-batches-py/data_batch_1', 'rb')
    D= None
    for i in xrange(m_num_iter):
        D = func(get_batch(train_set_x))

    # generate 256(8-by-16) images
    image = Image.fromarray( tile_raster_images(X=np.transpose(D), img_shape=(12, 12), tile_shape=(8, 16), tile_spacing=(1, 1)))
    image.save('repflds_minibatch.png')