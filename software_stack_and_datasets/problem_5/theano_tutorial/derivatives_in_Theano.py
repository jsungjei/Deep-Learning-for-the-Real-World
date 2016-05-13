# Derivatives in Theano ###############################################################################################
print "\n##### Derivatives in Theano #####"
# Computing Gradients -------------------------------------------------------------------------------------------------
print "\n----- Computing Gradients -----"

import numpy
import theano
import theano.tensor as T
from theano import pp
x = T.dscalar('x')
y = x ** 2
gy = T.grad(y, x)
print pp(gy)                           # print out the gradient prior to optimization
                                       #'((fill((x ** TensorConstant{2}), TensorConstant{1.0})
                                       # * TensorConstant{2}) * (x ** (TensorConstant{2} - TensorConstant{1})))'
f = theano.function([x], gy)
print f(4)                             #array(8.0)
print numpy.allclose(f(94.2), 188.4)   #True


x = T.dmatrix('x')
s = T.sum(1 / (1 + T.exp(-x)))
gs = T.grad(s, x)
dlogistic = theano.function([x], gs)
print dlogistic([[0, 1], [-1, -2]])          # array([[ 0.25      ,  0.19661193],
                                             #        [ 0.19661193,  0.10499359]])

# Computing the Jacobian ----------------------------------------------------------------------------------------------
print "\n----- Computing the Jacobian -----"
x = T.dvector('x')
y = x ** 2
J, updates = theano.scan(lambda i, y,x : T.grad(y[i], x),
                         sequences=T.arange(y.shape[0]), # is the list of Theano variables or dictionaries
                                                         # describing the sequences scan has to iterate over.
                         non_sequences=[y,x])            # is the list of arguments that are passed to fn at each steps.

f = theano.function([x], J, updates=updates)
print f([4, 4])                              # array([[ 8.,  0.],
                                             #        [ 0.,  8.]])

print f([4, 4, 4])

# Computing the Hessian -----------------------------------------------------------------------------------------------
print "\n----- Computing the Hessian -----"
x = T.dvector('x')
y = x ** 2
cost = y.sum()
gy = T.grad(cost, x)
H, updates = theano.scan(lambda i, gy,x : T.grad(gy[i], x),
                         sequences=T.arange(gy.shape[0]),
                         non_sequences=[gy, x])
f = theano.function([x], H, updates=updates)
print f([4, 4])                              # array([[ 2.,  0.],
                                             #        [ 0.,  2.]])

# Jacobian times a Vector ---------------------------------------------------------------------------------------------
print "\n----- Jacobian times a Vector -----"
# R-operator ----------------------------------------------------------------------------------------------------------
print "\n----- R-operator -----"

W = T.dmatrix('W')
V = T.dmatrix('V')
x = T.dvector('x')
y = T.dot(x, W)
JV = T.Rop(y, W, V)
f = theano.function([W, V, x], JV)
print f([[1, 1], [1, 1]], [[2, 2], [2, 2]], [0,1])      # array([ 2.,  2.])
# L-operator ----------------------------------------------------------------------------------------------------------
print "\n----- L-operator -----"
W = T.dmatrix('W')
v = T.dvector('v')
x = T.dvector('x')
y = T.dot(x, W)
VJ = T.Lop(y, W, v)
f = theano.function([v,x], VJ)
print f([2, 2], [0, 1])                                 # array([[ 0.,  0.],
                                                        # [ 2.,  2.]])

# Hessian times a Vector ----------------------------------------------------------------------------------------------
print "\n----- Hessian times a Vector -----"

x = T.dvector('x')
v = T.dvector('v')
y = T.sum(x ** 2)
gy = T.grad(y, x)
vH = T.grad(T.sum(gy * v), x)
f = theano.function([x, v], vH)
print f([4, 4], [2, 2])                                 # array([ 4.,  4.])


x = T.dvector('x')
v = T.dvector('v')
y = T.sum(x ** 2)
gy = T.grad(y, x)
Hv = T.Rop(gy, x, v)
f = theano.function([x, v], Hv)
print f([4, 4], [2, 2])                                 # array([ 4.,  4.])