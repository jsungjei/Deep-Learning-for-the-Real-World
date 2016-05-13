# Indexing ############################################################################################################
import numpy as np


# Assignment vs referencing ###########################################################################################
# Single element indexing #############################################################################################
print "\n##### Single element indexing #####"
x = np.arange(10)
print x
print x[2]
print x[-2]

x.shape = (2,5) # now x is 2-dimensional
print x
print x[1,3]
print x[1,-1]

print x[0]
print x[0][2]

# Other indexing options ##############################################################################################
print "\n##### Other indexing options  #####"
x = np.arange(10)
print "x"
print x
print "x[2:5]"
print x[2:5]
print "x[:-7]"
print x[:-7]
print "x[1:7:2]"
print x[1:7:2]
y = np.arange(35).reshape(5,7)
print "y"
print y
print "y[1:5:2,::3]"
print y[1:5:2,::3]
print "y[0:5:2,::3]"
print y[0:5:2,::3]
print "y[1:5:3,::3]"
print y[1:5:3,::3]

# Index arrays ########################################################################################################
print "\n##### Index arrays  #####"

x = np.arange(10,1,-1)
print x

print x[np.array([3, 3, 1, 8])]


print x[np.array([3,3,-3,8])]

# segment fault
# print x[np.array([3, 3, 20, 8])]

print x[np.array([[1,1],[2,3]])]

# Indexing Multi-dimensional arrays ###################################################################################
print "\n##### Indexing Multi-dimensional arrays  #####"

print "y[np.array([0,2,4]), np.array([0,1,2])]"
print y[np.array([0,2,4]), np.array([0,1,2])]

# shape mismatch
# print y[np.array([0,2,4]), np.array([0,1])]
print "y[np.array([0,2,4]), 1]"
print y[np.array([0,2,4]), 1]

print "y[np.array([0,2,4])]"
print y[np.array([0,2,4])]

# Boolean or mask index arrays ########################################################################################
print "\n##### Indexing Multi-dimensional arrays  #####"

b = y>20
print b
print "\ny[b]"
print y[b]

print "\nb[:,5]"
print b[:,5] # use a 1-D boolean whose first dim agrees with the first dim of y

print "\ny[b[:,5]]"
print y[b[:,5]]

x = np.arange(30).reshape(2,3,5)
print "\nx"
print x

b = np.array([[True, True, False], [False, True, True]])
print "\nb"
print b
print x[b]

# Combining index arrays with slices ##################################################################################
print "\n##### Combining index arrays with slices  #####"

print y[np.array([0,2,4]),1:3]

#print y[b[:,2],1:3]

# Structural indexing tools ###########################################################################################
print "\n##### Structural indexing tools  #####"
print "\ny.shape"
print y.shape
print "\ny"
print y
print "\ny[:,np.newaxis,:].shape"
print y[:,np.newaxis,:].shape

x = np.arange(5)
print "\nx"
print x

print "\nx[:,np.newaxis]"
print x[:,np.newaxis]

print "\nx[np.newaxis,:]"
print x[np.newaxis,:]

print "\nx[:,np.newaxis] + x[np.newaxis,:]"
print x[:,np.newaxis] + x[np.newaxis,:]

z = np.arange(81).reshape(3,3,3,3)
print "\nz"
print z

print "\nz[1,...,2]"
print z[1,...,2]

print "\nz[1,:,:,2]"
print z[1,:,:,2]
#print z[1,1,1,2]

# Assigning values to indexed arrays ##################################################################################
print "\n##### Assigning values to indexed arrays  #####"

x = np.arange(10)
print x

x[2:7] = 1
print x

x[2:7] = np.arange(5)
print x

x[1] = 1.2
print x

#x[1] = 1.2j
#print x

x = np.arange(0, 50, 10)
print x

x[np.array([1, 1, 3, 1])] += 1
print x

# Dealing with variable numbers of indices within programs ############################################################
print "\n##### Dealing with variable numbers of indices within programs  #####"

indices = (1,1,1,1)
print "\nz[indices], indices = (1,1,1,1)"
print z[indices]

indices = (1,1,1,slice(0,2)) # same as [1,1,1,0:2]
print "\nz[indices], indices = (1,1,1,slice(0,2))"
print z[indices]

indices = (1, Ellipsis, 1) # same as [1,...,1]
print "\nz[indices], indices = (1, Ellipsis, 1)"
print z[indices]

print "\nz[[1,1,1,1]]"
print z[[1,1,1,1]] # produces a large array

print "\nz[(1,1,1,1)]"
print z[(1,1,1,1)] # returns a single value