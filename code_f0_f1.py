# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 11:12:27 2017

@author: ccai
"""

def f(n=2):
    if n == 0:
        return 10
    elif n == 1:
        return 10
    elif n > 1:
        return f(n-1) + f(n-2)
    else:
        print('n less than 0', n)

def f1(n):
    if n < 2:
        return 10
    s= [10, 10, 0]    
    for i in range(2, n + 1):
        s[2] = s[1] + s[0]
        s[1], s[0] = s[2], s[1]
    return s[-1]
        
    
#%%
for i in range(5):    
    print(i, f(i), f1(i))