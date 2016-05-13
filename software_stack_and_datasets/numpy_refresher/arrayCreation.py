import numpy as np

# Converting Python array_like Objects to Numpy Arrays

print np.array([2,3,1,0])

print np.array([2, 3, 1, 0])
print np.array([[1,2.0],[0,0],(1+1j,3.)]) # note mix of tuple and lists,

print np.array([[ 1.+0.j, 2.+0.j], [ 0.+0.j, 0.+0.j], [ 1.+1.j, 3.+0.j]])

# Intrinsic Numpy Array Creation

print np.arange(10)

print np.arange(2, 10, dtype=np.float)

print np.arange(2, 3, 0.1)


print np.linspace(1., 4., 6)

print np.indices((3,3))

