# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 10:37:47 2017

@author: yaohaiying
"""

from sys import argv

script, filename = argv

txt = open(filename)

print "Here's your file %r:" % filename
print txt.read()

print "Type the filename again:"
file_again = raw_input("> ")

txt_again = open(file_again)

print txt_again.read()
