
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

x0 = 0.8
t = 0.
T = .1

fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(10) :
	tt = np.linspace(t,t+T,1000)
	J = HJacobi_integrator(H,None)

	if i == 0 :
		Jt = J(tt,lambda x : 100*(x-x0)**2 )
	else:
		Jt = J(tt,Js)

	xx = J.xx
	dx = xx[1]-xx[0]
	jst = H.seperatrix(xx)
	def dJdt(J,t=0):
		dJdx = np.gradient(J,dx)
		w = exp_cdf(t)
		V = (dJdx - jst)**2
		return - ((1-w)*H(xx,dJdx) + w*V)

	if i == 0 :
		sol = odeint(dJdt,100*(xx-x0)**2,tt)
	else:
		sol = odeint(dJdt,Js(xx),tt)
	
	ax.plot(xx,H.seperatrix(xx),'k-.')
	#ax.plot(J.xx,np.gradient(Jt,xx[1]-xx[0]),'b-')
	ax.plot(xx,np.gradient(sol[-1,],dx),'r-')

	Js = UnivariateSpline(xx,sol[-1,],s=0.1)
#	ax.plot(xx,np.gradient(Js(xx),dx))

	ax.set_ylim((-5.,5.))

	t += T

print exp_cdf(tt[-1])

fig2 = plt.figure()
ax = fig2.add_subplot(111)
for i in range(sol.shape[0]) :
	if i % 50 == 0 :
		ax.plot(xx,sol[i,])
ax.set_ylim((0.,10))

plt.show()
