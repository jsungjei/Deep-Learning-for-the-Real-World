# Structured arrays ###################################################################################################
import numpy as np
# Introduction ########################################################################################################
print "\n##### Introduction #####"

x = np.array([(1, 2., 'Hello'), (2, 3., "World")],
             dtype=[('foo', 'i4'), ('bar', 'f4'), ('baz', 'S10')])
print x

print x[1]

y = x['foo']
print y

y[:] = 2*y
print y

print x

x[1] = (-1,-1.,"Master")
print x

print y

# Defining Structured Arrays ##########################################################################################
# a) b1, i1, i2, i4, i8, u1, u2, u4, u8, f2, f4, f8, c8, c16, a<n>
#    (representing bytes, ints, unsigned ints, floats, complex and
#     fixed length strings of specified byte lengths)
# b) int8,...,uint8,...,float16, float32, float64, complex64, complex128
#    (this time with bit sizes)
# c) older Numeric/numarray type specifications (e.g. Float32).
#    Don't use these in new code!
# d) Single character type specifiers (e.g H for unsigned short ints).
#    Avoid using these unless you must. Details can be found in the
#    Numpy book

print "\n##### Defining Structured Arrays #####"

x = np.zeros(3, dtype='3int8, float32, (2,3)float64')
print x


x = np.zeros(3, dtype=('i4',[('r','u1'), ('g','u1'), ('b','u1'), ('a','u1')]))
print "\nx = np.zeros(3, dtype=('i4',[('r','u1'), ('g','u1'), ('b','u1'), ('a','u1')]))"
print x

print "\nx['r']"
print x['r']

x = np.zeros(3, dtype={'names':['col1', 'col2'], 'formats':['i4','f4']})
print "\nx = np.zeros(3, dtype={'names':['col1', 'col2'], 'formats':['i4','f4']})"
print x

x = np.zeros(3, dtype={'col1':('i1',0,'title 1'), 'col2':('f4',1,'title 2')})
print "\nx = np.zeros(3, dtype={'col1':('i1',0,'title 1'), 'col2':('f4',1,'title 2')})"
print x



# Accessing and modifying field names #################################################################################
print "\n##### Accessing and modifying field names #####"

print x.dtype.names

x.dtype.names = ('x', 'y')
print x.dtype.names


#x.dtype.names = ('x', 'y', 'z') # wrong number of names

print "\n##### Accessing field titles #####"

print x.dtype.fields['x'][2]

print "\n##### Accessing multiple fields at once #####"

x = np.array([(1.5,2.5,(1.0,2.0)),(3.,4.,(4.,5.)),(1.,3.,(2.,6.))],
             dtype=[('x', 'f4'), ('y', np.float32), ('value', 'f4', (2, 2))])

print x[['x', 'y']]

print x[['x', 'value']]


print x[['y', 'x']]

print "\n##### Filling structured arrays #####"

arr = np.zeros((5,), dtype=[('var1','f8'),('var2','f8')])
print arr
arr['var1'] = np.arange(5)
print arr['var1']
print arr

arr[0] = (10,20)
print arr

print "\n##### Record Arrays#####"

recordarr = np.rec.array([(1, 2., 'Hello'), (2, 3., "World")],
                         dtype=[('foo', 'i4'), ('bar', 'f4'), ('baz', 'S10')])

print recordarr

print "\nfoo"
print recordarr.foo
print "\nbar"
print recordarr.bar
print "\nbaz"
print recordarr.baz
print "\nrecordarr[0:3]"
print recordarr[0:3]
print "\nrecordarr[1:1]"
print recordarr[1:1]
print "\nrecordarr[1:2]"
print recordarr[1:2]
print "\nrecordarr[1:2].foo"
print recordarr[1:2].foo
print "\nrecordarr.foo[1:2]"
print recordarr.foo[1:2]
print "\nrecordarr[1].baz"
print recordarr[1].baz

# arr = array([(1, 2., 'Hello'), (2, 3., "World")], dtype=[('foo', 'i4'), ('bar', 'f4'), ('baz', 'S10')])
#
# recordarr = np.rec.array(arr)
#
# arr = np.array([(1, 2., 'Hello'), (2, 3., "World")],
#                dtype=[('foo', 'i4'), ('bar', 'f4'), ('baz', 'a10')])
# recordarr = arr.view(dtype=dtype((np.record, arr.dtype)),
#                      type=np.recarray)
#
#
# recordarr = arr.view(np.recarray)
# recordarr.dtype
#
#
# arr2 = recordarr.view(recordarr.dtype.fields or recordarr.dtype, np.ndarray)
#
# recordarr = np.rec.array([('Hello', (1, 2)), ("World", (3, 4))],
#                          dtype=[('foo', 'S6'), ('bar', [('A', int), ('B', int)])])
# type(recordarr.foo)
#
# type(recordarr.bar)


