
# http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity
# http://vipl.ict.ac.cn/sites/default/files/papers/files/2014_ACCV_Representation%20Learning%20with%20Smooth%20Autoencoder.pdf

import gzip
import cPickle
import matplotlib.pyplot as plt
import numpy
import time

import theano
import theano.tensor as T

class load_data(object):
    def __init__(self):
        '''
        Constructor
        '''

    def load_cifar_data(self):
        fo = open("cifar-10-batches-py/data_batch_1", 'rb')
        data = cPickle.load(fo)
        fo.close()
        return data

    def load_mnist_data(self):
        f = gzip.open("mnist.pkl.gz", 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close();

        def split_dataset(data_xy, borrow=True):
            data_x, data_y = data_xy

            return data_x, data_y

        test_set_x, test_set_y = split_dataset(test_set)
        valid_set_x, valid_set_y = split_dataset(valid_set)
        train_set_x, train_set_y = split_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        return rval

    def load_shared_mnist_data(self,dataset='mnist.pkl.gz'):
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        def shared_dataset(data_xy, borrow=True):
            data_x, data_y = data_xy
            shared_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),  # @UndefinedVariable
                                     borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),  # @UndefinedVariable
                                     borrow=borrow)
            return shared_x, T.cast(shared_y, 'int32')

        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        return rval

class SparseAutoEncoder(object):
    def __init__(self, num_input, num_hidden, batch_size=20, learning_rate=0.01, sparsity_lambda=0.01, n_epochs=100, reconstruction_cost="cross_entropy", sparsity_cost = "kl"):
        self.input = T.matrix(name='input')
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.sparsity_lambda = sparsity_lambda
        self.reconstruction_cost = reconstruction_cost
        self.sparsity_cost = sparsity_cost

        rand = numpy.random.RandomState(int(time.time()))
        r = -4 * numpy.sqrt(6. / (num_hidden + num_input + 1))

        self.W = theano.shared(
            value=numpy.asarray(
                rand.uniform(
                    low = -r,
                    high = r,
                    size=(num_input, num_hidden)
                ),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
                value=numpy.zeros(
                    num_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
        )

        self.c = theano.shared(
                value=numpy.zeros(
                    num_input,
                    dtype=theano.config.floatX
                ),
                name='c',
                borrow=True
        )

        self.params = [self.W, self.b, self.c]

        hidden_layer = self.encoder(self.input)
        output_layer = self.decoder(hidden_layer)

        self.encoder_f = theano.function(
            [self.input],
            output_layer
        )

    def encoder(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def decoder(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W.T) + self.c)

    # for binary inputs
    # cross-entropy (more precisely: sum of BErnoulli cross-entropies)
    def get_cross_entropy_loss_cost(self):
        x = self.input
        h = self.encoder(x)
        reproduced_x = self.decoder(h)
        L = (-T.sum(x * T.log(reproduced_x) + (1 - x) * T.log(1 - reproduced_x), axis=1))
        cost = T.mean(L);
        return cost

    # for real-valued inputs
    # sum of square differences (squared euclidean distance)
    def get_squared_diff_loss_cost(self):
        x = self.input
        h = self.encoder(x)
        reproduced_x= self.decoder(h)
        L = T.sum((reproduced_x-x)**2/2, axis=1)
        cost = T.mean(L);
        return cost

    def get_L1_cost(self):
        x = self.input
        h = self.encoder(x)
        L = T.sum(abs(h), axis=1)
        cost = T.mean(L);
        return  cost

    def kl_divergence(self):
        x = self.input
        y = self.encoder(x)

        # typically a sparsity parameter is a small value close to zero (say sparsity_param = 0.05)
        # rho
        sparsity_param = 0.05
        # rho_hat
        average_act_hidden_unit = y
        kl =T.sum(sparsity_param * T.log(sparsity_param/average_act_hidden_unit) + (1 - sparsity_param) * T.log((1-sparsity_param)/(1-average_act_hidden_unit)),axis=1)

        return T.mean(kl)

    def encode(self,inpt):
        out=self.encoder_f(inpt)
        return out

    def sparse_auto_encoding(self):
        data_loader = load_data()
        mnist_dataset = 'mnist.pkl.gz'
        [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
         (test_set_x, test_set_y)] = data_loader.load_shared_mnist_data(mnist_dataset)

        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // self.batch_size

        index = T.lscalar()
        x = self.input

        if self.reconstruction_cost == "cross_entropy":
            main_cost = self.get_cross_entropy_loss_cost()
        elif self.reconstruction_cost == "square_difference":
            main_cost = self.get_squared_diff_loss_cost()

        if self.sparsity_cost == "kl":
            # Kullback-Leibler(KL) divergence
            sparsity_cost = self.kl_divergence()
        elif self.sparsity_cost == "l1_cost":
            sparsity_cost = self.get_L1_cost()

        # TODO: try with different value of sparsity lambda
        # objective function of SpAE(Sparse autoencoder)
        obj_SpAE = main_cost + self.sparsity_lambda * sparsity_cost

        gparams = T.grad(obj_SpAE, self.params)

        updates = [
            (param, param - self.learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
            ]

        train_model = theano.function(
            [index],
            obj_SpAE,
            updates=updates,
            givens={
                x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        for epoch in xrange(self.n_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(train_model(batch_index))

            print ('cost: ' + str(numpy.mean(c)) + 'at epoch: ' + str(epoch))

def visualize_reconstruction(train_set_x, filename):
    fig, plots = plt.subplots(10, 10)
    for i in xrange(10):
        for j in xrange(10):
            index = j +  10 * i
            data = train_set_x[index,:]
            data = data - data.min()
            data = data / data.max()
            img = data.reshape(28, 28).astype('float32')

            plots[i, j].get_xaxis().set_visible(False)
            plots[i, j].get_yaxis().set_visible(False)
            plots[i, j].imshow(img)

    plt.set_cmap('binary')
    plt.savefig(filename)


if __name__ == '__main__':
    m_num_hidden = 300
    m_sparsity_lambda = 0.0001
    SqAE = SparseAutoEncoder(num_input=28 * 28, num_hidden=m_num_hidden, sparsity_lambda=m_sparsity_lambda, sparsity_cost= "kl")
    data_loader = load_data()
    [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
     (test_set_x, test_set_y)] = data_loader.load_mnist_data()

    SqAE.sparse_auto_encoding()

    # first 100 samples from the test set of MNIST
    autoencoderrec_figure_name = str(m_sparsity_lambda) + "_kl_autoencoderrec.png"
    visualize_reconstruction(SqAE.encode(test_set_x[:100]), autoencoderrec_figure_name)
    visualize_reconstruction(SqAE.encode(test_set_x[:100]), autoencoderrec_figure_name)
