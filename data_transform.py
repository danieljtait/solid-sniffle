# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:59:35 2016

@author: danieltait
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

X = np.loadtxt("dbwData1.txt",delimiter=",")
n = X.size
X = X[:,None]
tt = np.linspace(0.,0.5*n,n)
"""
Random Choice
""" 
class rRepSample:
    def __init__(self,X,h,seed):
        self.X = X
        self.h = h 
        self.seed = seed
        self.kd = KDTree(self.X)
    def make(self):
        np.random.seed(self.seed)
        Choices = range(self.X.shape[0]-2)
        Ivals = []
        Ireps = []
        while Choices != []:
            i = np.random.choice(Choices)
            B = self.kd.query_ball_point(self.X[i,],self.h)
            for b in B :
                if b in Choices:
                    Choices.remove(b)
            Ivals.append(i)
            Ireps.append(B)
        self.inds = Ivals
        self.Ireps = Ireps
        print "Original sample size is: ",self.X.shape[0]
        print "Effective sample size is: ",len(self.inds)

#dRep = rRepSample(X,h,91)
#dRep.make()        

