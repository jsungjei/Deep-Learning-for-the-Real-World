##PCA and sparse autoencoder

#Problem 20
Implement PCA in Python using Theano. Write your own implementation from ground up.<br />
**problem_20.py**<br />
<br />

#Problem 21
Produce a PCA scatterplot (see http://peekaboo-vision.blogspot.de/2012/12/another-look-at-mnist.html) on MNIST, also do this on CIFAR-10. Write them to file scatterplotMNIST.png and scatterplotCIFAR.png respectively.<br />
Code: **problem_21.py**<br />
Result: **scatterplotMNIST.png** and **scatterplotCIFAR.png**<br />
<br />
In order to use one of the data sets, MNIST or Cifar-10, please uncomment highlighted box in the code.<br />
It generates scatterplotMNIST.png or scatterplotCIFAR.png depending on the data set.<br />
<br />


#Problem 22
Implement an autoencoder in Python using Theano. Train the network using the squared error loss L(x) = (||f(x) − x||_2)^2 where x is a data sample and f(x) is the output of the autoencoder. You may follow a part of the tutorial or, better, write your own implementation from ground up. If training is difficult using gradient descent, try using the RMSprop optimizer.<br />
**problem_22.py**<br />
<br />
It generates folder dA_plots

#Problem 23
Increase the number of hidden units, but add a sparsity constraint on the hidden units. This means that the network should be encouraged to have most hidden units close to zero for a sample from the data set. This can be done by adding an L1 penalty (see literature section for reasons behind this) to the loss function, 
for example L_sparse(x) = L(x) + λ(|h(x)| _1) where h(x) denotes the hidden layer values for sample x
and |z|_1 = |z _i| _i is the L1 -norm of z. λ > 0 is a new hyperparameter that determines
the trade-off between sparsity and reconstruction error.<br />
**problem_23.py**<br />
<br />
number of hidden units and sparsity factor can be adjusted by changing below variables<br />
**m_num_hidden = 300**<br />
**m_sparsity_lambda = 0.0001**<br />



#Problem 24
Train the sparse autoencoder on MNIST. Write the reconstructions (i.e. outputs of the autoencoder) of the first 100 samples from the test set of MNIST into file autoencoderrec.png. Adjust λ and see how it affects the reconstructions.<br />
Code: **problem_24.py**<br />
Result: **problem_24_plots**<br />
<br />

Depending on the value of the sparsity factor the name of the output image will be different.<br />
image file name format: **[value of sparsity factor]_autoencoderrec.png**<br />

1. Setting different value of λ gives different cost<br />
2. Smaller λ brings good features <br />

#Problem 25
Code: **problem_25.py**<br />
Result: **problem_25_plots**<br />
<br />
number of hidden units and sparsity factor can be adjusted by changing below variables<br />
**m_num_hidden = 300**<br />
**m_sparsity_lambda = 0.0001**<br />

Depending on the value of the sparsity factor the name of the output image will be different.<br />
image file name format: **[value of sparsity factor]_autoencoderfilter.png**<br />

1. Setting different value of λ gives different cost<br />
2. Smaller λ brings good features <br />

#Problem 26
**problem_26.odt**<br />
<br />

#Bonus question 3
**bonus_3.py**<br />
<br />
In order to change the sparsity cost function, please replace the prameters like below<br />
"""<br />
sparsity cost = kl (--> Kullback-Leibler(KL) divergence), l1_cost (--> L1 cost)<br />
"""<br />
Line 241: SqAE = SparseAutoEncoder(num input=28 * 28, num_hidden = m num hidden, sparsity lambda = m sparsity lambda, sparsity cost= "kl")<br />

#Theoretic prerequisites
principal component analysis (PCA), autoencoder, sparsity

#Literature
Bishop: section 12.1,<br />
Murphy: sections 28.3.2, 28.3.3, 28.4.2, 28.4.3<br />
http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity<br />
http://stats.stackexchange.com/questions/45643/why-l1-norm-for-sparse-models<br />

#Tutorial
http://deeplearning.net/tutorial/dA.html#autoencoders

#Reference
MNIST data set: http://deeplearning.net/data/mnist/mnist.pkl.gz<br />
................http://deeplearning.net/tutorial/gettingstarted.html<br />
CIFAR-10 data set: http://www.cs.toronto.edu/~kriz/cifar.html<br />
