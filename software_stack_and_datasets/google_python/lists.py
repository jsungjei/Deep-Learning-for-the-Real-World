colors = ['red', 'blue', 'green']
print colors[0]                 ## red
print colors[2]                 ## green
print len(colors)               ## 3

b = colors                      ## Does not copy the list
                                ## array b just points the first address of the color array


## for and in

squares = [1, 4, 9, 16]
sum = 0
for num in squares:
    sum += num
print sum                       ## 30

list = ['larry', 'curly', 'moe']
if 'curly' in list:
    print 'yay'


## range

## print the numbers from 0 through 99
for i in range(5):
    print i


## while loop

## Access every 3rd element in a list with increment 3
print 'while loop test'
a = [1, 4, 9, 16]
i = 0
while i < len(a):
    print a[i]
    i = i + 3

list = ['larry', 'curly', 'moe']
list.append('shemp')  ## append elem at end
list.insert(0, 'xxx')  ## insert elem at index 0
list.extend(['yyy', 'zzz'])  ## add list of elems at end
print list  ## ['xxx', 'larry', 'curly', 'moe', 'shemp', 'yyy', 'zzz']
print list.index('curly')  ## 2

list.remove('curly')  ## search and remove that element
list.pop(1)  ## removes and returns 'larry'
print list  ## ['xxx', 'moe', 'shemp', 'yyy', 'zzz']

list = [1, 2, 3]
print list.append(4)  ## NO, does not work, append() returns None
## Correct pattern:
list.append(4)
print list  ## [1, 2, 3, 4]