# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 09:50:58 2016

@author: danieltait
"""

from Jintegrator import *

delta = [0.25,0.25]
scale = 1.5
dxTarg = 0.01

parInt = integrator_pars(delta,scale,dxTarg)
parInt.construct_xx(0.3)

H = makeH()

Jint = JIntegrator(parInt,H)


def j0f(xx):
    return 100*(xx-0.3)**2
    
tt = np.linspace(0.,1.5,15)

j0 = j0f(parInt.xx)

xx = parInt.xx

#plt.plot(xx,Jint.dJdt(j0))

z = Jint(j0,tt)
if type(z) == np.ndarray:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Jint.intPar.xx,np.exp(-2*z))
    
from main import pTransition2

pT = pTransition2(H,parInt)