import numpy
import theano.tensor as T
from theano import function

# Adding two Scalars ##################################################################################################
print "\n##### Adding two Scalars #####"

x = T.dscalar('x')
print x
y = T.dscalar('y')
print y
z = x + y

f = function([x, y], z)
print f(2, 3)
print numpy.allclose(f(16.3, 12.1), 28.4)

# step 1 --------------------------------------------------------------------------------------------------------------
print "\n----- step 1 -----"
x = T.dscalar('x')
y = T.dscalar('y')

print type(x)
print x.type
print T.dscalar
print x.type is T.dscalar

# step 2 --------------------------------------------------------------------------------------------------------------
print "\n----- step 2 -----"

from theano import pp
z = x + y

print(pp(z))

# step 3 --------------------------------------------------------------------------------------------------------------
print "\n----- step 3 -----"

f = function([x, y], z)

# Adding two Matrices #################################################################################################
print "\n##### Adding two Matrices #####"
x = T.dmatrix('x')
print x
y = T.dmatrix('y')
print y
z = x + y
f = function([x, y], z)
print f([[1, 2], [3, 4]], [[10, 20], [30, 40]])

# The following types are available:
#
# byte: bscalar, bvector, bmatrix, brow, bcol, btensor3, btensor4
# 16-bit integers: wscalar, wvector, wmatrix, wrow, wcol, wtensor3, wtensor4
# 32-bit integers: iscalar, ivector, imatrix, irow, icol, itensor3, itensor4
# 64-bit integers: lscalar, lvector, lmatrix, lrow, lcol, ltensor3, ltensor4
# float: fscalar, fvector, fmatrix, frow, fcol, ftensor3, ftensor4
# double: dscalar, dvector, dmatrix, drow, dcol, dtensor3, dtensor4
# complex: cscalar, cvector, cmatrix, crow, ccol, ctensor3, ctensor4
# Exercise ############################################################################################################
print "\n##### Exercise #####"
import theano

a = theano.tensor.vector()      # declare variable
out = a + a ** 10               # build symbolic expression
f1 = theano.function([a], out)   # compile function
print(f1([0, 1, 2]))

b =theano.tensor.vector()      # declare variable
out = a ** 2 + b ** 2 + 2 * a * b
f2 = theano.function([a, b], out)   # compile function
print(f2([1, 2], [3, 4]))