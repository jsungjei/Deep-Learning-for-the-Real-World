# Scan ################################################################################################################
print "\n##### Scan #####"
#
# - A general form of recurrence, which can be used for looping.
# - Reduction and map (loop over the leading dimensions) are special cases of scan.
# - You scan a function along some input sequence, producing an output at each time-step.
# - The function can see the previous K time-steps of your function.
# - sum() could be computed by scanning the z + x(i) function over a list, given an initial state of z=0.
# - Often a for loop can be expressed as a scan() operation, and scan is the closest that Theano comes to looping.
# - Advantages of using scan over for loops:
#    = Number of iterations to be part of the symbolic graph.
#    = Minimizes GPU transfers (if GPU is involved).
#    = Computes gradients through sequential steps.
#    = Slightly faster than using a for loop in Python with a compiled Theano function.
#    = Can lower the overall memory usage by detecting the actual amount of memory needed.

import theano
import theano.tensor as T
# Scan Example: A^K ---------------------------------------------------------------------------------------------------
print "\n----- Scan Example: A^K -----"
# There are three things here that we need to handle:
# the initial value assigned to result, the accumulation of results in result, and the unchanging variable A.
# Unchanging variables are passed to scan as non_sequences.
# Initialization occurs in outputs_info, and the accumulation happens automatically.

k = T.iscalar("k")
A = T.vector("A")

# Symbolic description of the result
result, updates = theano.scan(fn=lambda prior_result, A: prior_result * A,
                              outputs_info=T.ones_like(A),  # Returns a tensor filled with 1s that has same shape as A.
                              non_sequences=A,          # is the list of arguments that are passed to fn at each steps.
                              n_steps=k)          # is the number of steps to iterate given as an int or Theano scalar.

# We only care about A**k, but scan has provided us with A**1 through A**k.
# Discard the values that we don't care about. Scan is smart enough to
# notice this and not waste memory saving them.
final_result = result[-1]

# compiled function that returns A**k
power = theano.function(inputs=[A,k], outputs=final_result, updates=updates)

print(power(range(10),2))
print(power(range(10),4))

# Scan Example: Computing tanh(x(t).dot(W) + b) elementwise -----------------------------------------------------------
print "\n----- Scan Example: Computing tanh(x(t).dot(W) + b) elementwise -----"

import numpy as np

# defining the tensor variables
X = T.matrix("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")

results, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym),
                               sequences=X)

compute_elementwise = theano.function(inputs=[X, W, b_sym],
                                      outputs=results)

# test values
x = np.eye(2, dtype=theano.config.floatX)
w = np.ones((2, 2), dtype=theano.config.floatX)
b = np.ones((2), dtype=theano.config.floatX)
b[1] = 2

print(compute_elementwise(x, w, b))

# comparison with numpy
print(np.tanh(x.dot(w) + b))