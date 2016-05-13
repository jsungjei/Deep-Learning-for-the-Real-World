# Importing data with genfromtxt ######################################################################################

import numpy as np
from StringIO import StringIO

# Splitting the lines into columns ####################################################################################
# The delimiter argument ##############################################################################################
print "\n##### The delimiter argument #####"
data = "1, 2, 3\n4, 5, 6"
print data
print np.genfromtxt(StringIO(data), delimiter=",")

data = "  1  2  3\n  4  5 67\n890123  4"
print data
print np.genfromtxt(StringIO(data), delimiter=3)


# The autostrip argument ##############################################################################################
print "\n##### The autostrip argument #####"
data = "123456789\n   4  7 9\n   4567 9"
print data
print np.genfromtxt(StringIO(data), delimiter=(4, 3, 2))

data = "1, abc , 2\n 3, xxx, 4"
print data
# Without autostrip
# http://stackoverflow.com/questions/14790130/dtypes-difference-between-s1-and-s2-in-python
print np.genfromtxt(StringIO(data), delimiter=",", dtype="|S5")

# check belows
# print np.genfromtxt(StringIO(data), delimiter=",", dtype="|S3")
# print np.genfromtxt(StringIO(data), delimiter=",", dtype="|S2")

# With autostrip
print np.genfromtxt(StringIO(data), delimiter=",", dtype="|S5", autostrip=True)

# The comments argument ###############################################################################################
print "\n##### The comments argument #####"
data = """#
# Skip me !
# Skip me too !
1, 2
3, 4
5, 6 #This is the third line of the data
7, 8
# And here comes the last line
9, 0
"""
print data
# http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.genfromtxt.html
print np.genfromtxt(StringIO(data), comments="#", delimiter=",")

# Skipping lines and choosing columns #################################################################################
# The skip_header and skip_footer arguments ###########################################################################
print "\n##### The skip_header and skip_footer arguments #####"
data = "\n".join(str(i) for i in range(10))
print data
print np.genfromtxt(StringIO(data),)

print np.genfromtxt(StringIO(data),
              skip_header=3, skip_footer=5)
# The usecols argument ################################################################################################
print "\n##### The usecols argument #####"
data = "1 2 3\n4 5 6"
print data
print np.genfromtxt(StringIO(data), usecols=(0, -1))
# same with previous line
# print np.genfromtxt(StringIO(data), usecols=(0, 2))

data = "1 2 3\n4 5 6"
print data
print np.genfromtxt(StringIO(data),
              names="a, b, c", usecols=("a", "c"))


print np.genfromtxt(StringIO(data),
              names="a, b, c", usecols=("a, c"))

# Choosing the data type ##############################################################################################
# Setting the names ###################################################################################################
# The names argument ##################################################################################################

print "\n##### The names argument #####"
data = StringIO("1 2 3\n 4 5 6")
np.genfromtxt(data, dtype=[(_, int) for _ in "abc"])
# array([(1, 2, 3), (4, 5, 6)],
#       dtype=[('a', '<i8'), ('b', '<i8'), ('c', '<i8')])


data = StringIO("1 2 3\n 4 5 6")
np.genfromtxt(data, names="A, B, C")
# array([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
#       dtype=[('A', '<f8'), ('B', '<f8'), ('C', '<f8')])

data = StringIO("So it goes\n#a b c\n1 2 3\n 4 5 6")
np.genfromtxt(data, skip_header=1, names=True)
# array([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
#       dtype=[('a', '<f8'), ('b', '<f8'), ('c', '<f8')])

data = StringIO("1 2 3\n 4 5 6")
ndtype=[('a',int), ('b', float), ('c', int)]
names = ["A", "B", "C"]
np.genfromtxt(data, names=names, dtype=ndtype)
# array([(1, 2.0, 3), (4, 5.0, 6)],
#       dtype=[('A', '<i8'), ('B', '<f8'), ('C', '<i8')])

# The defaultfmt argument #############################################################################################
print "\n##### The defaultfmt argument #####"
data = StringIO("1 2 3\n 4 5 6")
print np.genfromtxt(data, dtype=(int, float, int))
# array([(1, 2.0, 3), (4, 5.0, 6)],
#       dtype=[('f0', '<i8'), ('f1', '<f8'), ('f2', '<i8')])

data = StringIO("1 2 3\n 4 5 6")
print np.genfromtxt(data, dtype=(int, float, int), names="a")
# array([(1, 2.0, 3), (4, 5.0, 6)],
#       dtype=[('a', '<i8'), ('f0', '<f8'), ('f1', '<i8')])

data = StringIO("1 2 3\n 4 5 6")
np.genfromtxt(data, dtype=(int, float, int), defaultfmt="var_%02i")
# array([(1, 2.0, 3), (4, 5.0, 6)],
#       dtype=[('var_00', '<i8'), ('var_01', '<f8'), ('var_02', '<i8')])


# Tweaking the conversion #############################################################################################
# The converters argument #############################################################################################
print "\n##### The converters argument #####"
convertfunc = lambda x: float(x.strip("%"))/100.
data = "1, 2.3%, 45.\n6, 78.9%, 0"
names = ("i", "p", "n")
# General case .....
print np.genfromtxt(StringIO(data), delimiter=",", names=names)
# array([(1.0, nan, 45.0), (6.0, nan, 0.0)],
#       dtype=[('i', '<f8'), ('p', '<f8'), ('n', '<f8')])


# Converted case ...
print np.genfromtxt(StringIO(data), delimiter=",", names=names,
              converters={1: convertfunc})
# array([(1.0, 0.023, 45.0), (6.0, 0.78900000000000003, 0.0)],
#       dtype=[('i', '<f8'), ('p', '<f8'), ('n', '<f8')])

# Using a name for the converter ...
print np.genfromtxt(StringIO(data), delimiter=",", names=names,
              converters={"p": convertfunc})
# array([(1.0, 0.023, 45.0), (6.0, 0.78900000000000003, 0.0)],
#       dtype=[('i', '<f8'), ('p', '<f8'), ('n', '<f8')])


data = "1, , 3\n 4, 5, 6"
convert = lambda x: float(x.strip() or -999)
print np.genfromtxt(StringIO(data), delimiter=",",
               converters={1: convert})
# array([[1., -999., 3.],
#        [4., 5., 6.]])


data = "N/A, 2, 3\n4, ,???"
kwargs = dict(delimiter=",",
              dtype=int,
              names="a,b,c",
              missing_values={0:"N/A", 'b':" ", 2:"???"},
              filling_values={0:0, 'b':0, 2:-999})
print np.genfromtxt(StringIO(data), **kwargs)
# array([(0, 2, 3), (4, 0, -999)],
#       dtype=[('a', '<i8'), ('b', '<i8'), ('c', '<i8')])


