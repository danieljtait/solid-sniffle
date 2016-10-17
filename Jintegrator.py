

import numpy as np 
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline
"""
Adaptive integration routine aiming to 
achieve stability and speed of numerical
integration of the Ham-Jacobi function

Uses linear extrapolation outside some range
"""
class integrator_pars:
	def __init__(self,delta=0.1,scale=1.25,dxTarget=0.01,N_max=100,ptol=1e-3):
		self.delta = delta # The left and right increments to change interval by
		self.scale = scale # 
		self.dxTarget=dxTarget #
		self.N_max = N_max
		self.ptol=ptol
		self.maxChanges = 10
		self.c = 0.3
		self.XLIM = [-1.5,1.5]
	def construct_xx(self,c):
		xl = max(c-self.scale*self.delta[0],self.XLIM[0])
		xu = min(c+self.scale*self.delta[1],self.XLIM[1])
		Nx = np.ceil( (xu-xl)/self.dxTarget )
		N = min(Nx,self.N_max)
		self.xx = np.linspace(xl,xu,N)
		self.dx = self.xx[1]-self.xx[0]
		return self.xx

def update_integrator_pars(changes,parIntObj):
	if changes[0] == 'Increase':
		parIntObj.delta[0]*=parIntObj.scale
	elif changes[0] == 'Decrease':
		parIntObj.delta[0]/=parIntObj.scale
	if changes[1] == 'Increase':
		parIntObj.delta[1]*=parIntObj.scale
	elif changes[1] == 'Decrease':
		parIntObj.delta[1]*=parIntObj.scale

def cond(Jt,integrator_parsObj):
	zL = np.exp(-2*Jt[0])
	zR = np.exp(-2*Jt[-1])
	#print "------------------"
	#print integrator_parsObj.dx
	#print integrator_parsObj.xx[0], integrator_parsObj.xx[-1]
	#print zL,zR
	lChange = None
	rChange = None
	noChange = True
	if zL > integrator_parsObj.ptol: 
		lChange = 'Increase'
		noChange = False
	if zR > integrator_parsObj.ptol:
		rChange = 'Increase'
		noChange = False
	return [lChange,rChange],noChange

def linExtrapolate(J0,intParObj):
	js = UnivariateSpline(intParObj.xx,J0,s=0.1)
	lrLeft = linregress(intParObj.xx[:3],js(intParObj.xx[:3]))
	lrRight = linregress(intParObj.xx[-3:],js(intParObj.xx[-3:]))
	xL = intParObj.xx[0]
	xR = intParObj.xx[-1]
	def f(xx):
		y = np.zeros(xx.size)
		for i in range(xx.size):
			x = xx[i]
			if x < xL :
				y[i] = lrLeft.slope*x + lrLeft.intercept
			elif x > xR :
				y[i] = lrRight.slope*x + lrRight.intercept
			else:
				y[i] = js(x)
		return y
	return f

from scipy.integrate import odeint
class JIntegrator:
	def __init__(self,intPar,HObj):
		self.intPar = intPar 
		self.H = HObj
	def dJdt(self,J,t=0):
		dJdx = np.gradient(J,self.intPar.dx)
		return -self.H(self.intPar.xx,dJdx)
	def __call__(self,J0,tt):
		nSteps = tt.size - 1
		Jcur = J0.copy()
		for i in range(nSteps):
			ttSmall = np.linspace(tt[i],tt[i+1],5)
			Jcur = self.adaptive_step(Jcur,ttSmall)
		return Jcur
	def adaptive_step(self,J0,ttSmall):
		nt = 0
		SUCCESS = False
#		import matplotlib.pyplot as plt 
		while nt < self.intPar.maxChanges:
			#print "nt is ",nt
			# Try integrator
#			fig = plt.figure()
#			ax = fig.add_subplot(111)
			sol,infodict = odeint(self.dJdt,J0,ttSmall,printmessg=1,full_output=True)
			nst = infodict['nst']
			if nst[3] > 1e5 :
				print "Something has gone really wrong"
			sol = sol[-1,]
#			ax.plot(self.intPar.xx,J0,'bo')
#			ax.plot(self.intPar.xx,sol,'r-')

			changes,noChange = cond(sol,self.intPar)
			# Pass solution to condition checker
			if noChange and abs(sol).max() < 1e100:
				#print "Took ",nt,"iterations."
				return sol 				
			else :
				# Construct the extrapolator of init cond
				# over regions it is currently definied
				jfunc = linExtrapolate(J0,self.intPar)
				# Carry out changes
				update_integrator_pars(changes,self.intPar)
				self.intPar.construct_xx(self.intPar.c)
				J0 = jfunc(self.intPar.xx)
#				ax.plot(self.intPar.xx,J0,'g+')
			nt += 1
		if not SUCCESS:
			print "Error: Failed to provide stable solution."
			return None

"""
Test it
"""
def makeH():
	import numpy as np 
	from main import init_dbw_example,potential,Hamiltonian,pStationary,HJacobi_integrator
	from main import pTransition
	from scipy.integrate import odeint
	from scipy.interpolate import UnivariateSpline

	
	x0 = 0.3
	T = 0.5
	par = 0.8

	D2 = .5

	def f(x):
		return -(-x**4 + 2*par*x**2)
	def fgrad(x):
		return -(4*x*(par-x**2))
	U = potential(f,fgrad)
	H = Hamiltonian(U,lambda x: D2)
	return H 

def test():
	import numpy as np 
	from main import init_dbw_example,potential,Hamiltonian,pStationary,HJacobi_integrator
	from main import pTransition
	from scipy.integrate import odeint
	from scipy.interpolate import UnivariateSpline

	
	x0 = 0.3
	T = 0.5
	par = 0.8

	D2 = .5

	def f(x):
		return -(-x**4 + 2*par*x**2)
	def fgrad(x):
		return -(4*x*(par-x**2))
	U = potential(f,fgrad)
	H = Hamiltonian(U,lambda x: D2)
	Pst = None
	J  = HJacobi_integrator(H,Pst)

	def J0f(x):
		return 10*(x-x0)**2

	import matplotlib.pyplot as plt 
	fig = plt.figure()
	ax = fig.add_subplot(111)

	def dJdt(J,t=0,xx=[1.,0.]):
		dx = xx[1]-xx[0]
		dJdx = np.gradient(J,dx)
		return -H(xx,dJdx)

