# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 10:46:37 2017

@author: yaohaiying
"""

class Song(object):
    
    def __init__(self,lyrics):
        self.lyrics = lyrics
        
    def sing_me_a_song(self):
        for line in self.lyrics:
            print line
            
happy_bday = Song(["Hppy birthday to you",
                   "I don't want to get sued",
                   "So I'll stop right there"])
                   
bulls_on_parade = Song(["They rally around the family",
                        "With pockets full of shells"])
                        
happy_bday.sing_me_a_song()

bulls_on_parade.sing_me_a_song()