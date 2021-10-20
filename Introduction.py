# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:20:13 2018

@author: Ueli
"""

# name convention
variable_name = 45  # variables small letters (no camelcase)
CONST_NAME = 324  # constants only capital letters


def function_name(parameter ):# function names same as variable names
    return 0  # intend is 4 spaces or tab


# basic data types
b = True  # Boolean
i = 123  # Integer
f = 1.34123  # float or double -> handled internally
q = -4.8e-5  # -4.8*10^-5
z = 2 + 3j  # complex number with imaginary unit j

text = 'Hello '  # String
text1 = "world"  # String
print(text + text1)

a = 10
b = 3
print(a // b)  # integer division
print(a % b)  # Modulo
print(a ** b)  # a^b

# lists

digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
empty = []  ## empty list

print(len(digits))  # length of list
print(digits[0])  # get element 0 from list

digits.append(10)  # add elemeent to the end of the list

digits.pop(6)  # remove 6th element of list
digits.remove(5)  # remove this is element from list. In this case all 5

digits.insert(5, 50)  # insert element at specific place

# list doesnt need to contain the same datatypes

# tuples - are like constant list, cant be changed

person = ("Ueli", 23, 1.83)

# unpacking

name, age, height = person

# dicts - tupels with name, elements are called key and value

person2 = {"name": "Oliver Knapp", "age": 23, "height": 1.83}
print(person2["age"])

person2["hair"] = "black"  # add this key value pair

# Control statements

a = 42
b = 42
c = 1
d = 0

print(a == b and d < c < b <= a)
print(a == b and d < c and b <= a)
print(not a == b or a == c)

# if statements

if a < b:
    print("a < b")
elif a == b:  # else if
    print(0)
elif a:  # if a is not 0 -> true, and if 0 false
    print("test")
else:
    print("hallo")

# while

i = 0
n = 10

while i < n:
    print(i)
    i += 1

# for loops - iterate over lists
list = ["a", "b", "c", "d"]

for letter in list:  # for each element do something
    print(letter)

for i in range(len(list)):  # "standard" for loop from other languages
    print(list[i])

for i, letter in enumerate(list):  # enumerate genreate a tuple with (index, element) of list
    print("list at %d: %s" % (i, letter))  # string formating


# functions

def say_hello():
    print("Hello world!")


say_hello()


def sum(a, b):
    return a + b


print(sum(3, 4))


# default values

def integer_division(a, b, return_reminder=False):
    if return_reminder:
        return a // b, a % b

    return a // b


print(integer_division(10, 3))
print(integer_division(10, 3, 1))
print(integer_division(return_reminder=True, a=10, b=3))


# *args and **kwargs
def sum(*args):
    sum = 0
    for number in args:
        sum += number
    return sum


print(sum(1, 2, 3, 4, 5))  # *args takes all arguments to a list


def echo(**kwargs):  # kwargs is a dictionary -> needs key value
    for key, value in kwargs.items():
        print("item with key %s is %s" % (key, value))


echo(name="Marcus", age=18, height=1)


# unpacking

def sum(a, b):
    return a + b


list_numbers = [3, 4]

print(sum(*list_numbers))  # * unpacks the values from the list


def check_right_trianlge(a, b, c):
    return a ** 2 + b ** 2 == c ** 2


big_triangle = {"a": 30, "b": 40, "c": 50}

print(check_right_trianlge(**big_triangle))

# lambda expressions

f = lambda x: 2 * x + 3  # Normal functions
g = lambda x: x ** 2


def compose(f1, f2, x):
    return f1(f2(x))


print(compose(f, g, 2))

# classes


class Person:
    def __init__(self, name, age, height):
        self.name = name
        self.age = age
        self.height = height


# packages

import numpy as np

a = [1, 4, 9, 16, 25]
a = np.array(a, dtype=np.float64)

print(a)

a = np.zeros((3, 4))  # 3x4 matrix with 0

print(a)

a = np.arange(5, 25, 2)  # like range but from numpy
print(a)

a = np.linspace(0, 1, 11)  # create lineare spaced list of values
print(a)

# slicing

a = np.arange(0, 12).reshape((4, 3))  # change shape of matrix
print(a)

print(a[2])

print(a[:, 2])

b = np.arange(12)
print(b)

print(b[3:10])
print(b[3:10:2])
print(b[::-1])

# broadcasting

print(a + 1)
print(a + [1, 2, 3])  # only functions if at least one dimension is matching

# vectorized operations

print(np.sin(a))  # calculates sin for each element

# reduction operators

print(np.sum(a))  # add all elements
print(np.sum(a, axis=0))  # add only rows, i.e. dimension 0
print(np.sum(a, axis=1))  # add only  columns

# condition indexing

print(a < 5)

print(a[a < 5])  # only print values that are smaller than 5

a[a >= 5] = 0
print(a)
