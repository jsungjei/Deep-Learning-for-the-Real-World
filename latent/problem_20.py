import cPickle
import gzip
import os
import numpy

# theano
import theano
import theano.tensor.nlinalg as linalg
import theano.tensor as T

# scikit-learn pca
from sklearn import decomposition

# plotting
import matplotlib.pyplot as plt

class LoadData(object):
    def __init__(self):
        '''
        constructor
        '''

    def load_mnist_data(self, mnist_dataset):
        f = gzip.open(mnist_dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        def shared_dataset(data_xy, borrow=True):
            data_x, data_y = data_xy
            shared_x = theano.shared(numpy.asarray(data_x[:10000],
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y[:10000],
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
            return shared_x, T.cast(shared_y, 'int32')

        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
        return rval

    def load_cifar_10_data(self, cifar_10_dataset):
        f = open(cifar_10_dataset, 'rb')
        data = cPickle.load(f)
        f.close()

        def shared_dataset(data, borrow=True):
            shared_data = theano.shared(numpy.asarray(data,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
            return shared_data

        shared_data = shared_dataset(data)

        return shared_data

class PCA(object):

    def __init__(self,  M):
        """
        :param M:   The new (reduced) dimension of the
                    data.
        """
        self.dim = M
        self.mean = None
        self.principal = None

    def dimemsion_transform(self, X):
        self.mean = T.mean(X, axis=0)
        X -= self.mean
        U, s, V = linalg.svd(X, full_matrices=False)

        self.principal = V[:self.dim]

        return linalg.matrix_dot(X, T.transpose(self.principal))


    def inverse_transform(self, X):
        """
        Perform an approximation of the input matrix of observations
        to the original dimensionality space

        :param X: The matrix of observations, where the training samples
        populate the rows, and the features populate the columns

        :return: Xhat, the dimensionality increased representation of the data
        """

        return linalg.matrix_dot(X, self.principal) + self.mean

if __name__ == '__main__':

    data_loader = LoadData()

    mnist_dataset = 'mnist.pkl.gz'
    # load the MNIST data
    [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)] = data_loader.load_mnist_data(mnist_dataset)

    # cifar_10_dataset = 'data_batch_1'
    # # load the cifar-10 data
    # cifar_10_data = data_loader.load_mnist_data(cifar_10_dataset)

    # Define the symbolic variables to be used
    X = T.matrix('X')
    Xtilde = T.matrix('Xtilde')

    # Initialize our model
    m_pca = PCA(100)

    # Theano function which fits the model to the
    # data i.e. applies dimensionality reduction
    transform = theano.function(
        inputs=[],
        outputs=m_pca.dimemsion_transform(X),
        givens={
            X: train_set_x
        }
    )

    # Apply the dimensionality reduction
    reduced_data = transform()

    # Theano function which approximates the
    # given data to the original dimensionality
    # on which the model was trained
    approximate = theano.function(
        inputs=[],
        outputs=m_pca.inverse_transform(Xtilde),
        givens={
            X: train_set_x,
            Xtilde: reduced_data
        }
    )

    Xhat = approximate()

    plt.matshow(Xhat[0,:].reshape((28,28)), cmap=plt.cm.gray)
    plt.matshow(Xhat[1, :].reshape((28, 28)), cmap=plt.cm.gray)

    # # compute PCA using scikit-learn for comparison
    # pca = decomposition.PCA(n_components=100)
    # pca.fit(train_set_x)
    # X = pca.transform(train_set_x)
    # X = pca.inverse_transform(X);
    # plt.matshow(X[0,:].reshape((28,28)), cmap=plt.cm.gray)

    plt.show()