s = 'hi'
print s[1]          ## i
print len(s)        ## 2
print s + ' there'  ## hi there

pi = 3.14
##text = 'The value of pi is ' + pi      ## NO, does not work
text = 'The value of pi is ' + str(pi)   ## yes
                                         ## str: built in function to change number to string

print text

## A "raw" string literal is prefixed by an 'r'
## and passes all the chars through
## without special treatment of backslashes,
## so r'x\nx' evaluates to the length-4 string
## 'x\nx'. A 'u' prefix allows you
## to write a unicode string literal
## (Python has lots of other unicode support features -- see the docs below).
raw = r'this\t\n and that'
print raw                                ## this\t\n and that

multi = """It was the best of times.
It was the worst of times."""            ## double double quotation marks allows newline

print multi


## NOTE: check for the other built in functions for string methods

## % operator
text = "%d little pigs come out or I'll %s and %s and %s" % (3, 'huff', 'puff', 'blow down')

print text

## other way to use % operator
## add parens to make the long-line work:
text = ("%d little pigs come out or I'll %s and %s and %s" %
        (3, 'huff', 'puff', 'blow down'))

print text

ustring = u'A unicode \u018e string \xf1'

print ustring

## (ustring from above contains a unicode string)
s = ustring.encode('utf-8')

print s

t = unicode(s, 'utf-8')             ## Convert bytes back to a unicode string

if t == ustring:
    print "true"
else:
    print "false"

## if-statement practice
speed = 90
mood = "terrible"

if speed >= 80:
    print 'License and registration please'
    if mood == 'terrible' or speed >= 100:
        print 'You have the right to remain silent.'
    elif mood == 'bad' or speed >= 90:
        print "I'm going to have to write you a ticket."
        write_ticket()
    else:
        print "Let's try to keep it under 80 ok?"