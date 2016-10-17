
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

z = np.sort(X)

def objFunc(par):
	# Set up the necessary functions
	def f(x):
		return -(-x**4 + 2*par*x**2)
	def fgrad(x):
		return -(4*x*(par-x**2))
	U = potential(f,fgrad)
	H = Hamiltonian(U,lambda x: D2)

	eps = 0.0
	def func(x,p):
		return eps*(p-H.seperatrix(x))**2
	H.set_add_term(func)

	Pst = None

	rootPar = np.sqrt(par)
	#xRep = [-0.63,0.63]
	xRep = [z[20],z[-20]]
	try :
		val = 0.
		J  = HJacobi_integrator(H,Pst)
		pT1 = pTransition(J)
		pT1.make(xRep[0],tt)
		pT2 = pTransition(J)
		pT2.make(xRep[1],tt)	

		xx = J.xx

		"""
		fig  = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(xx,pT1(xx))
		ax.plot(xx,pT2(xx))
		"""
		val = 0.
		for i in range(X.size-1):
			x = X[i]
			xT = X[i+1]
			if x < xRep[0] :
				val += np.log(pT1(xT))
			elif x > xRep[1] :
				val += np.log(pT2(xT))
			else:
				w = abs(x-xRep[0])/(xRep[1]-xRep[0])
				val += np.log( w*pT1(xT) + (1-w)*pT2(xT) )
		return -val
	except:
		return np.inf
print z[20],z[-20]

ll = []
pars = np.linspace(0.6,0.81,15)
for p in pars:
	ll.append(objFunc(p))
ll=np.array(ll)
print pars[np.where(ll == ll.min())[0]]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(pars,ll)

from scipy.optimize import minimize
res = minimize(objFunc,[0.7],method='Nelder-Mead',options={ 'disp': True , 'xatol' :1e-2})

print res
plt.show()

