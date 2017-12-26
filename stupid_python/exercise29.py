# -*- coding: utf-8 -*-
"""
Created on Wed May 03 14:04:14 2017

@author: yaohaiying
"""

people = 20
cats = 30
dogs = 15

if people < cats:
    print "Too many cats! The world is doomed!"

if people > cats:
    print "Not many cats! The world is saved!"
    
if people < dogs:
    print "The world is drooled on!"
    
dogs += 5

if people >= dogs:
    print "People are greater than or equal to dogs."
    
if people <= dogs:
    print "People are less than or equal to dogs."
    
if people == dogs:
    print "People are dogs."