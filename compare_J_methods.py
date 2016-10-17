
import numpy as np 
import matplotlib.pyplot as plt 
from main import init_dbw_example,potential,Hamiltonian,pStationary,HJacobi_integrator
from main import pTransition, pTransition2
from JIntegrator import integrator_pars
from scipy.interpolate import UnivariateSpline


X = np.loadtxt('dbwData1.txt',delimiter=',')
T = 2.1
tt = np.linspace(0.,T,1000)
D2 = .5

par = .8

def f(x):
	return -(-x**4 + 2*par*x**2)
def fgrad(x):
	return -(4*x*(par-x**2))
U = potential(f,fgrad)
H = Hamiltonian(U,lambda x: D2)

eps = 0.01
def func(x,p):
	return eps*(p-H.seperatrix(x))**2
H.set_add_term(func)

Pst = None

J  = HJacobi_integrator(H,Pst)



from scipy.integrate import odeint
from scipy.misc import derivative
x0 = 0.1
print derivative(lambda x: fgrad(x), x0=x0,dx=1e-6)
def dXdt(X,t=0):
	v= -fgrad(X[0])
	return np.array([v])
mode = odeint(dXdt,x0,np.linspace(0.,T,1000))
print "The mode is: ",mode[-1], np.sqrt(par)



pT1 = pTransition(J)
pT1.make(x0,tt)


intPar = integrator_pars(delta=[0.3,0.3],scale=1.5,dxTarget=0.01)

pT2 = pTransition2(H,intPar)
pT2.make(x0,T)

xx = np.linspace(-1.5,1.5,1000)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xx,pT1(xx),'b-')
ax.plot(xx,pT2(xx),'r-')	

plt.show()