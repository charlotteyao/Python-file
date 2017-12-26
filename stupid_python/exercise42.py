# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 16:32:05 2017

@author: yaohaiying
"""

## Animal is-a object (yes, sort of confusing) look at the extra credit
class Animal(object):
    pass

## ??
class Dog(Animal):
    
    def __init__(self,name):
        ## ??
        self.name = name

## ??
class Cat(Animal):
    
    def __init__(self,name):
        ## ??
        self.name = name
        
## ??
class Person(object):
    
    def __init__(self,name):
        ## ??
        self.name = name
        
        ##Person has-a pet of some kind
        self.pet = None
        
## ??
class Employee(Person):
    
    def __init__(self,name,salary):
        ## ?? hmm what is this strange magic?
        super(Employee,self).__init__(name)
        ## ??
        self.salary = salary

## ??
class Fish(object):
    pass

## ??
class Salmon(Fish):
    pass

## ??
class Halibut(Fish):
    pass

## rover is-a Dog
rover = Dog("Rover")

## satan is-a cat
satan = Cat("Saran")

## mary is-a person
mary = Person("Mary")

## ??