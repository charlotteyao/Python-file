# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:01:15 2017

@author: yaohaiying
"""

tabby_cat = "\tI'm tabbed in."
persian_cat = "I'm spliot\n on a line."
backslash_cat = "I'm \\ a \\ cat."

fat_cat = """
I'll do a list:
\t* Cat food
\t* Fishies
\t* Catnip\n\t* Grass
"""

print tabby_cat
print persian_cat
print backslash_cat
print fat_cat


while True:
    for i in ["/","-","|","\\","|"]:
        print "%s\r" % i