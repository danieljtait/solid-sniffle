
import numpy as np 

def hit_sp(x0,T,HObj):
	def dVdt(X,t=0):
		return H.D1(X[0])
	tau = 1e-3
	t = 0
	STATIONARY = False
	x = x0
	xprev = 2*x0 
	while t < T and not STATIONARY:
		v = dVdt([x],t)
		if abs(v) < 1e-3 :
			STATIONARY = True
			break
		elif abs(x-xprev) < 1e-5 :
			STATIONARY = True
			break
		xprev = x 
		x += v*tau
		t+= tau
	return x,t,STATIONARY


def get_Jt(x0,T,HObj,wfunc) :

	def dJdt(J,t,dx,xx,jst) :
		dJdx = np.gradient(J,dx)
		w = wfunc(t)
		V = (dJdx - jst)**2
		return -((1-w)*HObj(xx,dJdx) + w*V)
	#fig = plt.figure()
	#ax = fig.add_subplot(111)
	#ax.set_ylim((-5.,5.))
	"""
	Declare grid
	"""
	xx = np.linspace(-1.5,1.5,500)
	dx = xx[1]-xx[0]
	jst = HObj.seperatrix(xx)
	#ax.plot(xx,jst,'k-.')
	"""
	Pick tStep less than time before sp of dist hit
	"""

	mode,tHit,hit_mode = hit_sp(x0,T,HObj)
	print mode,tHit,hit_mode 
	if hit_mode:
		tStep = 0.75*tHit
	else:
		tStep = 0.75*T
	#tStep = 0.1
	t = 0.
	tt = np.linspace(t,tStep,1000*tStep)
	"""
	Carry out first integration, check tStep < T 
	"""
	Jt = odeint(dJdt,100*(xx-x0)**2,tt,args=(dx,xx,jst))[-1,]
	t += tStep 
	#ax.plot(xx,np.gradient(Jt,dx))
	if tStep > T :
		# Do once to T and done
		return Jt
	else:
		while t < T :
			#Smooth current
			tt = np.linspace(t,min(T,t+tStep),1000*tStep)
			Jspline = UnivariateSpline(xx,Jt,s=0.1)
			Jt = odeint(dJdt,Jspline(xx),tt,args=(dx,xx,jst))[-1,]
			#ax.plot(xx,np.gradient(Jt,dx))
			t+=tStep
			if abs(t-T) < 1e-6:
				break
		return Jt,xx



import numpy as np 
import matplotlib.pyplot as plt 
from main import init_dbw_example,potential,Hamiltonian,pStationary,HJacobi_integrator
from main import pTransition, pTransition2
from JIntegrator import integrator_pars
from scipy.interpolate import UnivariateSpline
from scipy.integrate import odeint
import matplotlib.pyplot as plt

D2 = .5

par = .8

def f(x):
	return -(-x**4 + 2*par*x**2)
def fgrad(x):
	return -(4*x*(par-x**2))
U = potential(f,fgrad)
H = Hamiltonian(U,lambda x: D2)

def exp_cdf(t,l = 1.3):
	return 1-np.exp(-l*t)

x0 = 1.3
t = 0.
T = .8

from scipy.integrate import quad

JT1,xx = get_Jt(0.8,T,H,exp_cdf)
JT2,_ = get_Jt(-0.03,T,H,exp_cdf)
JT3,_ = get_Jt(-0.8,T,H,exp_cdf)

JT4,_ = get_Jt(0.5,T,H,exp_cdf)

js1 = UnivariateSpline(xx,JT1,s=0.1)
js2 = UnivariateSpline(xx,JT2,s=0.1)
js3 = UnivariateSpline(xx,JT3,s=0.1)
js4 = UnivariateSpline(xx,JT4,s=0.1)

I1,err = quad(lambda x:np.exp(-js1(x)/H.D2(x)) , -2., 2.)
I2,err = quad(lambda x:np.exp(-js2(x)/H.D2(x)) , -2., 2.)
I3,err = quad(lambda x:np.exp(-js3(x)/H.D2(x)) , -2., 2.)
I4,err = quad(lambda x:np.exp(-js4(x)/H.D2(x)) , -2., 2.)

class MyLikelihood:
	def __init__(self,initPoints,pTs):
		self.initPoints = np.array(initPoints)
		self.initPoints = np.sort(self.initPoints)
		self.Np = self.initPoints.size
		self.pTs = pTs 
	def evalp(self,x0,xT):
		ind = np.sum(self.initPoints <= x0)
		if ind == 0 :
			return self.pTs[0](xT)
		elif ind == self.Np:
			return self.pTs[-1](xT)
		else:
			w = np.abs(x0-self.initPoints[ind-1])/abs(self.initPoints[ind]-self.initPoints[ind-1])
			return (1-w)*self.pTs[ind-1](xT) + w*self.pTs[ind](xT)

	# Sort x0
	def evalp2(self,x0,xT):
		ind = np.sum(self.initPoints <= x0)
		if ind == 0 :
			return self.pTs[str(self.initPoints[0])](xT)
		elif ind == self.Np:
			return self.pTs[str(self.initPoints[-1])](xT)
		else:
			p1 = self.pTs[str(self.initPoints[ind-1])]
			p2 = self.pTs[str(self.initPoints[ind])]
			w = np.abs(x0-self.initPoints[ind-1])/abs(self.initPoints[ind]-self.initPoints[ind-1])
			return (1-w)*p1(xT) + w*p2(xT)

	def addP(self,x0,pT):
		self.initPoints = np.sort(np.concatenate((self.initPoints,[x0])))
		self.Np += 1
		self.pTs[str(x0)] = pT

def PT1(x) :
	return np.exp(-js1(x)/H.D2(x))/I1
def PT2(x) :
	return np.exp(-js2(x)/H.D2(x))/I2
def PT3(x) :
	return np.exp(-js3(x)/H.D2(x))/I3
def PT4(x) :
	return np.exp(-js4(x)/H.D2(x))/I4

pDict = { str(0.8): PT1,
		  str(-0.03): PT2,
		  str(-0.8): PT3 }

LL = MyLikelihood([-0.03,-0.8,0.8],pDict)

x0val = 0.8
x0s = np.linspace(-0.03,0.8,200)
y = np.ones(x0s.size)*x0val
z1 = [LL.evalp2(x,x0val) for x in x0s]


LL.addP(0.5,PT4)

#z = np.zeros(x0s.size)
z2 = [LL.evalp2(x,x0val) for x in x0s]
#for i in range(x0s.size):
#	w = abs(x0s[i]-(-0.03))/((0.8)-(-0.03))
#	z[i] = (1-w)*q1 + w*q2
#	print w,q1,q2,z[i]

print "-------------"
print LL.evalp2(0.5,0.8), PT4(0.8)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot(xx,0.8*np.ones(xx.size),PT1(xx),'k-')
ax.plot(xx,-0.03*np.ones(xx.size),PT2(xx),'k-')
#ax.plot(xx,-0.8*np.ones(xx.size),PT3(xx),'k-')
ax.plot(xx,0.5*np.ones(xx.size),np.exp(-JT4/H.D2(xx))/I4)
ax.set_ylim((-0.05,1.))
ax.plot(y,x0s,z1,'k-')
ax.plot(y,x0s,z2,'k-')


plt.show()