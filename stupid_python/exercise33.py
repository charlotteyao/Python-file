# -*- coding: utf-8 -*-
"""
Created on Thu May 04 10:52:45 2017

@author: yaohaiying
"""

i = 0
numbers = []

while i < 6:
    print "At the top i is %d" % i
    numbers.append(i)
    
    i = i + 1
    print "Numbers now: ",numbers
    print "At the bottom i is %d" % i
    
print "The numbers: "

for num in numbers:
    print num