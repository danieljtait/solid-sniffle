"""
Contains the source code for the classes

Potential 

Hamiltonian

HJacobi_Integrator

pStationary

pTransition

"""
import numpy as np 
from scipy.integrate import odeint,quad
from scipy.stats import norm
def init_dbw_example():
	def f(x):
		return -(-x**4 + 2*0.8*x**2)
	def fgrad(x):
		return -(4*x*(0.8-x**2))
	U = potential(f,fgrad)
	return f,fgrad
"""
Potential Function Class
"""
class potential:
	def __init__(self,evalf,evalfgrad=None):
		self.evalf = evalf
		self.evalfgrad = evalfgrad
	def __call__(self,x):
		return self.evalf(x)
	def grad(self,x):
		return self.evalfgrad(x)

"""
The Hamiltonian - from potential
"""
class Hamiltonian:
	def __init__(self,potentialObj,D2):
		self.D1 = lambda x : -1*potentialObj.grad(x)
		self.D2 = D2
		self.add_term = False
	def __call__(self,x,p):
		addVal = 0.
		if self.add_term:
			addVal = self.add_term_ev(x,p)
		return 0.5*self.D2(x)*p**2 + p*self.D1(x) + addVal
	def seperatrix(self,x):
		return -2*self.D1(x)/self.D2(x)
	def set_add_term(self,func):
		self.add_term = True
		self.add_term_ev = func


class pStationary:
	def __init__(self,HamiltonianObj):
		self.H = HamiltonianObj
		self.NConstant = self.get_norm_constant()
	def J(self,xx):
		xx = np.asarray(xx)
		if xx.size == 1 :
			return quad(self.H.seperatrix,0.,xx)[0]
		else:
			return np.array([quad(self.H.seperatrix,0.,x)[0] for x in xx])
	def get_norm_constant(self):
		ans,err = quad(lambda x: np.exp(-self.J(x)) ,-np.inf,np.inf)
		return ans
	def __call__(self,x):
		return np.exp(-self.J(x))/self.NConstant

class HJacobi_integrator:
	def __init__(self,HamiltonianObj,pStat=None):
		self.H = HamiltonianObj
		self.mesh_set = False
		self.pStat = pStat
	"""
	wfunc(t) returns a number between 0,1
	wfunc(0) = 0
	wfunc(t) -> 1 as t-> inf
	"""
	def set_homotope_wfunc(self,wFunc):
		self.homotope_wFunc = wFunc 
	"""
	targFunc(x) returns the value x is homotoped too
	"""
	def set_homotope_target(self,targFunc):
		self.homotope_targFunc = targFunc
	"""
	set the mesh
	"""
	def set_mesh(self,xlim,Nx):
		self.xlim = xlim
		self.Nx = Nx
		self.xx = np.linspace(xlim[0],xlim[1],Nx)
		self.dx = self.xx[1]-self.xx[0]
		self.mesh_set = True
	def dJdt(self,J,t=0):
		dJdx = np.gradient(J,self.dx)
		return -self.H(self.xx,dJdx)
	def dJdt_cflowhomotope(self,J,t=0):
		dJdx = np.gradient(J,self.dx)
		w = self.homotope_wFunc(t)
		p = (1.-w)*dJdx + w*self.homotopeTarget
		return -self.H(self.xx,p)
	def __call__(self,tt,J0f,method="ordinary"):
		self.set_mesh([-1.5,1.5],500)
		if method == "ordinary":
			j0 = J0f(self.xx)
			sol = odeint(self.dJdt,j0,tt)
			return sol[-1,]
		elif method == "cflowhomotope" :
			w = util_cflowhomotope(1.5)
			self.set_homotope_wfunc(w)
			self.set_homotope_target(self.H.seperatrix)
			self.homotopeTarget = self.homotope_targFunc(self.xx)
			j0 = J0f(self.xx)
			sol = odeint(self.dJdt_cflowhomotope,j0,tt)
			return sol[-1,]
		elif method == "cflow2":
			#fig = plt.figure()
			ax = fig.add_subplot(111)
			#wfunc = util_cflowhomotope(.1)
			wfunc = util_cflowhomotope_2(.7)
			j0 = J0f(self.xx)
			T = tt[-1]
			N = tt.size
			NSteps = 2
			delT = T/NSteps
			tcur = 0.
			Jcur = j0.copy()
			J_stat = self.pStat.J(self.xx)
			ax.plot(self.xx,np.exp(-Jcur))
			for i in range(NSteps):
				ttSmall = np.linspace(tcur,tcur+delT,N/NSteps)
				"""
				Solve the ordinary flow
				"""
				sol = odeint(self.dJdt,Jcur,ttSmall)
				Jt = sol[-1,]
				"""
				Smoothly homotope to new solution
				"""
				tcur += delT
				w = wfunc(tcur)
				Jcur = (1-w)*Jt + w*J_stat
				ax.plot(self.xx,np.exp(-Jcur))
			return Jcur
		else:
			print "Not a valid method."

def util_cflowhomotope(rho=1.):
	def w(t):
		return 2*(1./(1.+np.exp(-rho*t)) - 0.5)
	return w

def util_cflowhomotope_2(m):
	def w(t) :
		if t < 1./m :
			return m*t 
		else :
			return 1.
	return w

"""
The whole sample will have one shared instance of 
HJacobi_integrator built with one H, and we then construct
pTransition
"""
from scipy.interpolate import UnivariateSpline
class pTransition:
	def __init__(self,HJacobi_integrator):
		self.HJacobi_integrator = HJacobi_integrator
	def make(self,x0,tt):
		def J0f(x):
			return 100*(x-x0)**2 
		Jt = self.HJacobi_integrator(tt,J0f,"ordinary")
		self.Jt_spline = UnivariateSpline(self.HJacobi_integrator.xx,Jt,s=0.1)
		I,err = quad(lambda x: np.exp(-2*self.Jt_spline(x)),-np.inf,np.inf)
		self.NConstant = I
	def __call__(self,x):
		return np.exp(-2*self.Jt_spline(x))/self.NConstant
"""
Version of pTransition using the more stable
JIntegrator
"""
from JIntegrator import JIntegrator, integrator_pars
class pTransition2:
	def __init__(self,HObj,integrator_pars):
		self.HJacobi_integrator = JIntegrator(HObj,integrator_pars)
	def make(self,x0,T,tsteps=0.1,parInt=None):
		# Form the initial interval around x0
		self.HJacobi_integrator.intPar.construct_xx(x0)
		# Initialise tsteps
		tt = np.arrange(0.,T,tsteps)
		# Initial steep quadratic
		def J0f(x):
			return 100*(x-x0)**2
		j0 = J0f(Self.HJacobi_integrator.intPar.xx)
		Jt = self.HJacobi_integrator(j0,tt)
		return Jt


"""

import copy
#Hnew = copy.deepcopy(H)

import matplotlib.pyplot as plt 

f,fg = init_dbw_example()
xx = np.linspace(-1.5,1.5,100)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xx,f(xx))
ax.plot(xx,-fg(xx))

U = potential(f,fg)
H = Hamiltonian(U,lambda x : 0.5)

fig2 = plt.figure()
ax = fig2.add_subplot(111)
x,p = np.meshgrid(xx,np.linspace(-5.,5.,xx.size))
Hvals = H(x.ravel(),p.ravel())
cs = ax.contour(x,p,Hvals.reshape(x.shape))

fig3 = plt.figure()
ax = fig3.add_subplot(111)
Pst = pStationary(H)
N = Pst.get_norm_constant()

ax.plot(xx,Pst(xx))

ans,err = quad(lambda x: Pst(x) , -np.inf,np.inf)
print ans

w = util_cflowhomotope(3.)
w2 = util_cflowhomotope_2(0.1)
fig4 = plt.figure()
ax = fig4.add_subplot(111)
tt = np.linspace(0.,15.,1000)
ax.plot(tt,w(tt))
ax.plot(tt,[w2(t) for t in tt])

"""

"""

fig5 = plt.figure()
ax = fig5.add_subplot(111)
def J0f(x):
	C = 100.
	return C*(x-0.3)**2
J = HJacobi_integrator(H,Pst)
tt = np.linspace(0.,.5,200)
Jt = J(tt,J0f,"cflow2")
#JTord = J(tt,J0f,"ordinary")
#JTcflowh = J(tt,J0f,"cflowhomotope")
#ax.plot(J.xx,J0f(xx))
#ax.plot(J.xx,np.exp(-JTord))
#ax.plot(J.xx,np.exp(-JTcflowh))
#ax.plot(J.xx,np.gradient(JTcflowh,J.dx))
#print H(J.xx,np.gradient(JTcflowh,J.dx))

class Diffusion:
	def __init__(self,HamiltonianObj):
		self.H = HamiltonianObj
	def sim(self,x0,T,n=100):
		delT = np.float(T)/n
		root_delT = np.sqrt(delT)
		S = [x0]
		for i in range(n-1):
			w = self.H.D2(S[-1])*norm.rvs(scale=root_delT)
			S.append( S[-1] + self.H.D1(S[-1])*delT + w )
		return np.array(S)
	def EstMoments(self,x0,T,nSim=50,n=100,withMix=False):
		np.random.seed(11)
		X = np.zeros(nSim)
		for i in range(nSim) :
			s = self.sim(x0,T,n)
			X[i] = s[-1]
		if withMix :
			from sklearn import mixture
			clf = mixture.GaussianMixture(n_components=2,covariance_type='full')
			clf.fit(X.reshape(nSim,1))
			return np.mean(X),np.var(X),clf
		else :
			return np.mean(X),np.var(X)


T = 0.62
x0 = 0.3

pT = pTransition(J)
pT.make(x0,T)


dX = Diffusion(H)
s = dX.sim(x0,T,100)
fig6 = plt.figure()
ax = fig6.add_subplot(111)
ax.plot(np.linspace(0.,T,100),s)

mu,var,clf = dX.EstMoments(x0,T,100,200,True)
print clf.means_
print clf.covariances_

fig7 = plt.figure()
ax = fig7.add_subplot(111)

from scipy.stats import multivariate_normal
def mixpdf(xx,clf) :
	val = 0.
	for i,w in enumerate(clf.weights_):
		val += w*multivariate_normal.pdf(xx,clf.means_[i],clf.covariances_[i])
	return val

xx = np.linspace(-1.5,1.5,100)
ax.plot(xx,pT(xx))
ax.plot(xx,mixpdf(xx,clf))

Mu,_ = quad(lambda x: x*pT(x), -np.inf,np.inf)
Var,_ = quad(lambda x: pT(x)*(x-Mu)**2, - np.inf,np.inf)

print Mu,Var

plt.show()
"""
