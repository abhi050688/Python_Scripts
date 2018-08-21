# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 12:44:52 2018

@author: Abhishek S
"""

import os
import numpy as np
import pandas as pd

os.chdir("E:/Deep Learning/Sequence Models/Week 2/mojify_w2_a2/data")
filename=os.listdir('.')[1]

def read_glove(filename):
    word2vec={}
    
    with open(filename,'r',encoding='utf8') as f:
        for i in f:
            line=i.split(' ')
            word2vec[str(line[0])]=np.asarray(line[1:])
    return word2vec
        
word2vec=read_glove(filename)    
word2vec['the'].shape
word2vec['run']
word2vec['stage']
word2vec["--"]
len(word2vec)

