import cPickle
import gzip
import numpy

import matplotlib.pyplot as plt
from itertools import product

import theano
import theano.tensor.nlinalg as linalg
# from scipy import linalg
import theano.tensor as T

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
            return data_x, data_y
            # shared_x = theano.shared(numpy.asarray(data_x,
            #                                    dtype=theano.config.floatX),
            #                      borrow=borrow)
            # shared_y = theano.shared(numpy.asarray(data_y,
            #                                    dtype=theano.config.floatX),
            #                      borrow=borrow)
            # return shared_x, T.cast(shared_y, 'int32')

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

        return data


class PCA(object):
    def __init__(self, M):
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
        return linalg.matrix_dot(X, self.principal) + self.mean

if __name__ == '__main__':

    data_loader = LoadData()

    ##### in order to mnist_dataset, please uncomment this box #############################################
    mnist_dataset = 'mnist.pkl.gz'                                                                        #
    [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)] = data_loader.load_mnist_data( mnist_dataset)                                                        #
    ########################################################################################################


    # #### in order to test cifar_10_dataset, please uncomment this box #####################################
    # cifar_10_dataset = data_loader.load_cifar_10_data('cifar-10-batches-py/data_batch_1')                                     #
    # cifar_10_label_names = data_loader.load_cifar_10_data('cifar-10-batches-py/batches.meta')                                 #
    #                                                                                                       #
    # train_set_x, train_set_y = cifar_10_dataset['data'] / 255., numpy.asarray(cifar_10_dataset['labels']) #
    # cifar_10_label_names =  cifar_10_label_names['label_names']                                           #
    # ########################################################################################################
    pca = PCA(2)

    X = T.matrix('X', dtype='float64')

    # Theano function which fits the model to the
    # data i.e. applies dimensionality reduction
    transform = theano.function(
        inputs=[X],
        outputs=pca.dimemsion_transform(X),
    )

    fig, plots = plt.subplots(10, 10)
    fig.set_size_inches(50, 50)
    plt.prism()
    for i, j in product(xrange(10), repeat=2):
        if i > j:
            continue

        X_ = train_set_x[(train_set_y == i) + (train_set_y == j)]
        y_ = train_set_y[(train_set_y == i) + (train_set_y == j)]

        X_transformed = transform(X_)

        plots[i, j].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
        plots[i, j].set_xticks(())
        plots[i, j].set_yticks(())

        plots[j, i].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
        plots[j, i].set_xticks(())
        plots[j, i].set_yticks(())
        if i == 0:
            ##### in order to test mnist_dataset, please uncomment this box ###########
            plots[i, j].set_title(j)                                                 #
            plots[j, i].set_ylabel(j)                                                #
            ###########################################################################
            # #### in order to test cifar_10_dataset, please uncomment this box #########
            # plots[i, j].set_title(cifar_10_label_names[j])                            #
            # plots[j, i].set_ylabel(cifar_10_label_names[j])                           #
            # ###########################################################################

    plt.tight_layout()
    #### in order to test cifar_10_dataset, please uncomment this box ####
    plt.savefig("scatterplotMNIST.png")                                  #
    ######################################################################

    # #### in order to test mnist_dataset, please uncomment this box #######
    # plt.savefig("scatterplotCIFAR.png")                                  #
    # ######################################################################

