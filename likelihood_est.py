

import numpy as np 
import matplotlib.pyplot as plt 
from main import init_dbw_example,potential,Hamiltonian,pStationary,HJacobi_integrator
from main import pTransition
from scipy.interpolate import UnivariateSpline


X = np.loadtxt('dbwData1.txt',delimiter=',')
T = 0.5
tt = np.linspace(0.,T,50)

# For now fix diffusion constant
D2 = .5

def objFuncMF(par):
	# Set up the necessary functions
	def f(x):
		return -(-x**4 + 2*par*x**2)
	def fgrad(x):
		return -(4*x*(par-x**2))
	U = potential(f,fgrad)
	H = Hamiltonian(U,lambda x: D2)
	Pst = pStationary(H)

	return -np.sum(np.log(Pst(X)))

def objFunc(par):
	# Set up the necessary functions
	def f(x):
		return -(-x**4 + 2*par*x**2)
	def fgrad(x):
		return -(4*x*(par-x**2))
	U = potential(f,fgrad)
	H = Hamiltonian(U,lambda x: D2)
	Pst = None

	try :
		val = 0.
		for i in range(X.size-1) :
			J  = HJacobi_integrator(H,Pst)
			pT = pTransition(J)
			pT.make(X[i],tt)
			val += -np.log(pT(X[i+1]))
		return val
	except:
		return np.inf

#import scipy
#res = scipy.optimize.minimize(objFunc,0.75,method='Nelder-Mead')
#print res

ll = []
pars = [0.6,0.65,0.7,.75,0.8,0.85,0.9]
pars = np.linspace(0.6,0.9,15)
for par in pars :
	val = objFunc(par)
	ll.append(val)
print "----------"
print ll

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.linspace(0.,T*X.size,X.size),X)

fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.plot(pars,ll)

plt.show()