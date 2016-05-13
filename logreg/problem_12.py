from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

import climin
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

try:
    import PIL.Image as Image
except ImportError:
    import Image

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
        # NOTE
        self.n_in = n_in
        # NOTE
        self.n_out = n_out

        self.params = theano.shared(
            value=numpy.zeros(
                n_in * n_out + n_out,
                dtype=theano.config.floatX
            ),
            name='params',
            borrow=True
        )

        self.W = self.params[0:n_in * n_out].reshape((n_in, n_out))

        self.B = self.params[n_in * n_out:n_in * n_out + n_out]

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.B)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.input = input

    def negative_log_likelihood(self, y, lmda):
        return -(T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])) + T.mean(0.3 * T.square(self.W))

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):

            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset):

    print('... loading data')

    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX), borrow=borrow)

        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval


def sgd_optimization_mnist_climin(optimizer_type, n_epochs=500,
                                  dataset='mnist.pkl.gz',
                                  batch_size=600):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print('... building the model')

    # index to minibatch
    index = T.lscalar()

    x = T.matrix('x')
    y = T.ivector('y')

    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    cost = classifier.negative_log_likelihood(y, 0.3).mean()

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index : index + batch_size],
            y: test_set_y[index : index + batch_size]
        },
        name="test"
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index : index + batch_size],
            y: valid_set_y[index : index + batch_size]
        },
        name="validate"
    )

    training_loss = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index : index + batch_size],
            y: train_set_y[index : index + batch_size]
        },
        name="training_loss"
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index: index + batch_size],
            y: valid_set_y[index: index + batch_size]
        },
        name="validate"
    )

    params_cost = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: train_set_x[index: index + batch_size],
            y: train_set_y[index: index + batch_size]
        },
        name="params_cost"
    )

    params_grad = theano.function(
        inputs=[index],
        outputs=T.grad(cost, classifier.params),
        givens={
            x: train_set_x[index: index + batch_size],
            y: train_set_y[index: index + batch_size]
        },
        name="params_grad"
    )

    def fnc(params):
        classifier.params.set_value(params, borrow=True)
        # An array containing the cost for each of the minibatches
        train_losses = [params_cost(i * batch_size)
                        for i in xrange(n_train_batches)]
        return numpy.mean(train_losses)

    def fnc_prime(params):
        classifier.params.set_value(params, borrow=True)
        grad = params_grad(0)
        for i in xrange(1, n_train_batches):
            grad += params_grad(i * batch_size)
        return grad / n_train_batches

    # creates the validation function
    def optimization(params, current_epoch):
        classifier.params.set_value(params, borrow=True)

        # compute the validation loss
        validation_losses = [validate_model(i * batch_size)
                             for i in xrange(n_valid_batches)]
        this_validation_loss = numpy.mean(validation_losses)
        validation_set_error.append(this_validation_loss * 100.)

        # compute the test loss
        test_losses = [test_model(i * batch_size)
                       for i in xrange(n_test_batches)]
        this_test_loss = numpy.mean(test_losses)
        test_set_error.append(this_test_loss * 100.)

        # compute the training loss
        training_losses = [training_loss(i * batch_size)
                           for i in xrange(n_train_batches)]
        this_train_loss = numpy.mean(training_losses)
        training_set_error.append(this_train_loss * 100.)

        print('validation error %f %% at epoch %i' % (this_validation_loss * 100.,current_epoch) )

        if this_validation_loss < validation_scores[0]:

            # improve patience if loss improvement is good enough
            if this_validation_loss < validation_scores[0] * \
                    improvement_threshold:
                patience[0] = max(patience[0], current_epoch * n_train_batches * patience[1])

            validation_scores[0] = this_validation_loss
            validation_scores[1] = numpy.mean(test_losses)
            validation_scores[2] = current_epoch
            validation_scores[3] = params

        if patience <= current_epoch * n_train_batches:
            return False
        else:
            return True

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')

    # early-stopping parameters
    # patience[0] = patience, patience[1] = patience_increase
    patience = [5000, 2]
    improvement_threshold = 0.995

    best_validation_loss = numpy.inf

    # Keeping track of training, testing and validation errors to plot error
    validation_set_error = []
    test_set_error = []
    training_set_error = []

    validation_scores = [best_validation_loss, 0, 0, None]

    ###############
    # TRAIN MODEL #
    ###############

    start_time = timeit.default_timer()

    # Select optimization method
    opt = None

    if optimizer_type == 'gd':
        opt = climin.GradientDescent(numpy.zeros((classifier.n_in + 1) * classifier.n_out, dtype=x.dtype), fnc_prime)
    elif optimizer_type == 'lbfgs':
        opt = climin.Lbfgs(numpy.zeros((classifier.n_in + 1) * classifier.n_out, dtype=x.dtype), fnc, fnc_prime)
    elif optimizer_type == 'ncg':
        opt = climin.NonlinearConjugateGradient(numpy.zeros((classifier.n_in + 1) * classifier.n_out, dtype=x.dtype),
                                                fnc, fnc_prime)
    elif optimizer_type == 'rmsprop':
        opt = climin.RmsProp(numpy.zeros((classifier.n_in + 1) * classifier.n_out, dtype=x.dtype), fnc, fnc_prime,
                             step_rate=1e-4, decay=0.9)
    elif optimizer_type == 'rprop':
        opt = climin.Rprop(numpy.zeros((classifier.n_in + 1) * classifier.n_out, dtype=x.dtype), fnc_prime)
    else:
        print ("unknown optimizer")
        return 1

    for info in opt:
        if (not optimization(opt.wrt, info['n_iter'])) or (info['n_iter'] >= n_epochs - 1):
            break

    end_time = timeit.default_timer()

    print(
        (
            'Optimization complete with best validation score of %f %%, with '
            'test performance %f %%'
        )
        % (validation_scores[0] * 100., validation_scores[1] * 100.)
    )
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

    # Plot the errors against the epochs
    epochs = numpy.arange(0, n_epochs)
    plt.plot(epochs, training_set_error, 'g', epochs, validation_set_error, 'r', epochs, test_set_error, 'b')
    plt.xlabel('epoch')
    plt.ylabel('error')

    # Create plot legend
    blue_patch = mpatches.Patch(color='green', label='train set error')
    green_patch = mpatches.Patch(color='red', label='validation set error')
    red_patch = mpatches.Patch(color='blue', label='test set error')
    best_validation_point, = plt.plot(validation_scores[2], validation_scores[0] * 100., 'o', mec='g', ms=15, mew=1, mfc='none',
                             label="best validation epoch")
    plt.legend(handles=[blue_patch, green_patch, red_patch, best_validation_point], numpoints=1)
    plt.savefig('error.png')

if __name__ == '__main__':
    ##
    # sgd_optimization_mnist_climin(<type of optimizer>)
    # <type of optimizer> : Gradient Descent                    -> gd
    #                       Quasi-Newton (BFGS, L-BFGS)         -> lbfgs
    #                       (non-linear) Conjugate Gradients    -> ncg
    #                       Resilient Propagation               -> rprop
    #                       rmsprop                             -> rmsprop
    sgd_optimization_mnist_climin(n_epochs=500, batch_size = 600, optimizer_type="lbfgs")
