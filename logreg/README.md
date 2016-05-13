##Multiclass logistic regression

#Problem 8

#Problem 9
Change parameters to generate different result<br />
"""<br />
learning_rate: learning rate<br />
n_epochs: number of epoches<br />
dataset: mnist dats set<br />
batch_size: batch size<br />
"""<br />
**line 475: sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,**<br />
...........................**dataset='mnist.pkl.gz',**<br />
...........................**batch_size=600)**<br />

#Problem 10
For Problem 10 to 13, Change parameters to generate different result<br />
"""<br />
sgd_optimization_mnist_climin(<type of optimizer>)<br />
type of optimizer : Gradient Descent                    -> gd<br />
....................Quasi-Newton (BFGS, L-BFGS)         -> lbfgs<br />
....................(non-linear) Conjugate Gradients    -> ncg<br />
....................Resilient Propagation               -> rprop<br />
....................rmsprop                             -> rmsprop<br />
"""<br />
**line 000: sgd_optimization_mnist_climin(n_epochs=500, batch_size=600, optimizer_type="lbfgs")**<br />
#Problem 11
it generate figures depending on the type of optimizer<br />
gd(GradientDescent) generate gd_repflds.png<br />
lbfgs(Quasi-Newton) generate lbfgs_repflds.png<br />
ncg(Conjugate Gradients) generate ncg_repflds.png<br />
rprop(Resilient Propagation) generate rprop_repflds.png<br />
rmsprop generate rmprop_repflds.png<br />

#Problem 12
it generate figures depending on the type of optimizer<br />
gd(GradientDescent) generate gd_error.png<br />
lbfgs(Quasi-Newton) generate lbfgs_error.png<br />
ncg(Conjugate Gradients) generate ncg_error.png<br />
rprop(Resilient Propagation) generate rprop_error.png<br />
rmsprop generate rmprop_error.png<br />

#Problem 13
it generate figures depending on the type of optimizer<br />
gd(GradientDescent) generate gd_error.png<br />
lbfgs(Quasi-Newton) generate lbfgs_error.png<br />
ncg(Conjugate Gradients) generate ncg_error.png<br />
rprop(Resilient Propagation) generate rprop_error.png<br />
rmsprop generate rmprop_error.png<br />

#Bonus question 1
Please open text file

#Theoretic prerequisites
logistic regression, optimisation (gradient descent, conjugate gradient),receptive field

#Literature
Bishop: section 4.3.2 – 4.3.4

#Tutorial
http://deeplearning.net/tutorial/logreg.html

#Reference
MNIST data set: http://deeplearning.net/data/mnist/mnist.pkl.gz<br />
................http://deeplearning.net/tutorial/gettingstarted.html<br />
CIFAR-10 data set: http://www.cs.toronto.edu/~kriz/cifar.html<br />
