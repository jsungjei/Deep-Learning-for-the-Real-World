a = [5, 1, 4, 3]
print sorted(a)                      ## [1, 3, 4, 5]
print a                              ## [5, 1, 4, 3]

strs = ['aa', 'BB', 'zz', 'CC']
print sorted(strs)                   ## ['BB', 'CC', 'aa', 'zz'] (case sensitive)
print sorted(strs, reverse=True)     ## ['zz', 'aa', 'CC', 'BB']

strs = ['ccc', 'aaaa', 'd', 'bb']
print sorted(strs, key=len)          ## ['d', 'bb', 'ccc', 'aaaa']

## "key" argument specifying str.lower function to use for sorting
print sorted(strs, key=str.lower)    ## ['aa', 'BB', 'CC', 'zz']

## Say we have a list of strings we want to sort by the last letter of the string.
strs = ['xc', 'zb', 'yd', 'wa']


## Write a little function that takes a string, and returns its last letter.
## This will be the key function (takes in 1 value, returns 1 value).
def MyFn(s):
    return s[-1]


print MyFn("ababc")

## Now pass key=MyFn to sorted() to sort by the last letter:
print sorted(strs, key=MyFn)         ## ['wa', 'zb', 'xc', 'yd']

#alist.sort()  ## correct
#alist = blist.sort()  ## NO incorrect, sort() returns None

tuple = (1, 2, 'hi')
print len(tuple)                     ## 3
print tuple[2]                       ## hi
#tuple[2] = 'bye'                     ## NO, tuples cannot be changed
tuple = (1, 2, 'bye')                ## this works

tuple = ('hi',)                      ## size-1 tuple

(x, y, z) = (42, 13, "hike")
print z                              ## hike

def Foo():
    s = ['abc', 'def']
    return s


(err_string, err_code) = Foo()       ## Foo() returns a length-2 tuple

print err_string
print err_code

nums = [1, 2, 3, 4]

squares = [n * n for n in nums]      ## [1, 4, 9, 16]

strs = ['hello', 'and', 'goodbye']

shouting = [s.upper() + '!!!' for s in strs]
## ['HELLO!!!', 'AND!!!', 'GOODBYE!!!']

## Select values <= 2
nums = [2, 8, 1, 6]
small = [n for n in nums if n <= 2]  ## [2, 1]

## Select fruits containing 'a', change to upper case
fruits = ['apple', 'cherry', 'bannana', 'lemon']
afruits = [s.upper() for s in fruits if 'a' in s]
## ['APPLE', 'BANNANA']