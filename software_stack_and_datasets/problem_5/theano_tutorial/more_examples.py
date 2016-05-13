# More Examples #######################################################################################################
print "\n##### More Examples #####"
# Logistic Functions --------------------------------------------------------------------------------------------------
print "\n----- Logistic Functions -----"

import theano
import theano.tensor as T
x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
# check the equations here, http://deeplearning.net/software/theano/tutorial/examples.html
logistic = theano.function([x], s)
print logistic([[0, 1], [-1, -2]])


s2 = (1 + T.tanh(x / 2)) / 2
logistic2 = theano.function([x], s2)
logistic2([[0, 1], [-1, -2]])

# Computing More than one Thing at the Same Time ----------------------------------------------------------------------
print "\n----- Computing More than one Thing at the Same Time -----"
a, b = T.dmatrices('a', 'b')
print a
print b
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
f = theano.function([a, b], [diff, abs_diff, diff_squared])

print f([[1, 1], [1, 1]], [[0, 1], [2, 3]])

# Setting a Default Value for an Argument -----------------------------------------------------------------------------
print "\n----- Setting a Default Value for an Argument -----"
from theano import In
from theano import function
x, y = T.dscalars('x', 'y')
z = x + y
f = function([x, In(y, value=1)], z)    # default value for the second argument is 1
print f(33)
print f(33, 2)

x, y, w = T.dscalars('x', 'y', 'w')
z = (x + y) * w
f = function([x, In(y, value=1), In(w, value=2, name='w_by_name')], z)
print f(33)
print f(33, 2)
print f(33, 0, 1)
print f(33, w_by_name=1)
print f(33, w_by_name=1, y=0)


# Using Shared Varialbes ----------------------------------------------------------------------------------------------
print "\n----- Using Shared Varialbes -----"
from theano import shared

state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])

print(state.get_value())
accumulator(1)
print(state.get_value())
accumulator(300)
print(state.get_value())
state.set_value(-1)
accumulator(3)
print(state.get_value())
decrementor = function([inc], state, updates=[(state, state-inc)])
decrementor(2)
print(state.get_value())

fn_of_state = state * 2 + inc
# The type of foo must match the shared variable we are replacing
# with the ``givens``
foo = T.scalar(dtype=state.dtype)
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
print skip_shared(1, 3)  # we're using 3 for the state, not state.value
print(state.get_value())  # old state still there, but we didn't use it


# Using Random Numbers ------------------------------------------------------------------------------------------------
print "\n----- Using Random Numbers -----"
# Brief Examples ------------------------------------------------------------------------------------------------
print "\n----- Brief Examples -----"

from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
srng = RandomStreams(seed=234)
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))

f = function([], rv_u)
g = function([], rv_n, no_default_updates=True)    #Not updating rv_n.rng
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)
print "\nnearly_zeros()"
print nearly_zeros()

f_val0 = f()
print "\nf_val0"
print f_val0
f_val1 = f()  # different numbers from f_val0
print "\nf_val1"
print f_val1

g_val0 = g()  # different numbers from f_val0 and f_val1
print "\ng_val0"
print g_val0
g_val1 = g()  # same numbers as g_val0!
print "\ng_val1"
print g_val1

nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)
print "\nnearly_zeros()"
print nearly_zeros()


# Seeding Streams -----------------------------------------------------------------------------------------------------
print "\n----- Seeding Streams -----"

rng_val = rv_u.rng.get_value(borrow=True)   # Get the rng for rv_u
rng_val.seed(89234)                         # seeds the generator
rv_u.rng.set_value(rng_val, borrow=True)    # Assign back seeded rng
srng.seed(902340)                           # seeds rv_u and rv_n with different seeds each

# Sharing Streams Between Functions -----------------------------------------------------------------------------------
print "\n----- Sharing Streams Between Functions -----"

state_after_v0 = rv_u.rng.get_value().get_state()
nearly_zeros()       # this affects rv_u's generator
print "\nnearly_zeros()"
print nearly_zeros()

v1 = f()
print "\nv1"
print v1
rng = rv_u.rng.get_value(borrow=True)
rng.set_state(state_after_v0)
rv_u.rng.set_value(rng, borrow=True)

v2 = f()             # v2 != v1
print "\nv2"
print v2

v3 = f()             # v3 == v1
print "\nv3"
print v3

# Copying Random State Between Theano Graphs --------------------------------------------------------------------------
print "\n----- Copying Random State Between Theano Graphs -----"

import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
class Graph():
        def __init__(self, seed=123):
            self.rng = RandomStreams(seed)
            self.y = self.rng.uniform(size=(1,))
g1 = Graph(seed=123)
f1 = theano.function([], g1.y)
g2 = Graph(seed=987)
f2 = theano.function([], g2.y)
# By default, the two functions are out of sync.
f1()
f2()

def copy_random_state(g1, g2):
    if isinstance(g1.rng, MRG_RandomStreams):
        g2.rng.rstate = g1.rng.rstate
    for (su1, su2) in zip(g1.rng.state_updates, g2.rng.state_updates):
        su2[0].set_value(su1[0].get_value())
# We now copy the state of the theano random number generators.
copy_random_state(g1, g2)
f1()

f2()


# A Real Example: Logistic Regression ---------------------------------------------------------------------------------
print "\n----- A Real Example: Logistic Regression -----"

import numpy
import theano
import theano.tensor as T
rng = numpy.random

N = 400                                   # training sample size
feats = 784                               # number of input variables

# generate a dataset: D = (input_values, target_class)
# http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.random.randint.html
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 10000

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")

# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)
w = theano.shared(rng.randn(feats), name="w")

# initialize the bias term
b = theano.shared(0., name="b")

print("Initial model:")
print(w.get_value())
print(b.get_value())

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # w.r.t weight vector w and
                                          # bias term b
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))