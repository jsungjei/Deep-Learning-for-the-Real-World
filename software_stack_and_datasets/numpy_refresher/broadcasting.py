# Indexing ############################################################################################################
import numpy as np

a = np.array([1.0, 2.0, 3.0])

b = np.array([2.0, 2.0, 2.0])
print a * b

a = np.array([1.0, 2.0, 3.0])
b = 2.0
print a * b

# General Broadcasting Rules ##########################################################################################
print "\n##### General Broadcasting Rules #####"

x = np.arange(4)
xx = x.reshape(4,1)
print "\nxx"
print xx
y = np.ones(5)
print "\ny"
print y
z = np.ones((3,4))
print "\nz"
print z

print "\nx.shape"
print x.shape

print "\ny.shape"
print y.shape

# <type 'exceptions.ValueError'>: shape mismatch: objects cannot be broadcast to a single shape
# x + y

print "\nxx.shape"
print xx.shape

print "\ny.shape"
print y.shape

print "\n(xx + y).shape"
print (xx + y).shape

print "\nxx + y"
print xx + y

print "\nx.shape"
print x.shape

print "\nz.shape"
print z.shape

print "\n(x + z).shape"
print (x + z).shape

print "\nx + z"
print x + z

a = np.array([0.0, 10.0, 20.0, 30.0])
print "\na"
print a
b = np.array([1.0, 2.0, 3.0])
print "\nb"
print b

print "\na[:, np.newaxis] + b"
print a[:, np.newaxis] + b

