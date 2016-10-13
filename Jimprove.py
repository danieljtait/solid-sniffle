import numpy as np 
from main import init_dbw_example,potential,Hamiltonian,pStationary,HJacobi_integrator
from main import pTransition
from scipy.integrate import odeint
from scipy.interpolate import UnivariateSpline


np.random.seed(11)

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


def dJdt(J,t=0,xx=None,dx=1.):
	dJdx = np.gradient(J,dx)
	return -H(xx,dJdx)

delta = 0.1
scale = 1.1
dxTarg = 0.01

class myExtrap:
    def __init__(self,xx,yy,xlim):
        # get gradients at left and
        # right ends
        self.xlim = xlim
        self.js = UnivariateSpline(xx,yy)
        self.lTail = xx[:3]
        self.lTail_par = linregress(self.lTail,self.js(self.lTail))
        self.rTail = xx[-3:]
        self.rTail_par = linregress(self.rTail,self.js(self.rTail))
    def __call__(self,x):
        if x < self.xlim[0]:
            return self.lTail_par.slope*x + self.lTail_par.intercept
        elif x >self. xlim[1]:
            return self.rTail_par.slope*x + self.rTail_par.intercept
        else:
            return self.js(x)
    
class integrator_pars:
	def __init__(self,delta,scale,dxTarget,tt):
		self.delta=delta
		self.scale=scale
		self.dxTarget=dxTarget
		self.tt=tt
	def construct_xx(self,xmode):
		xl = xmode-self.scale*self.delta
		xu = xmode+self.scale*self.delta
		Nx = np.ceil( (xu-xl)/self.dxTarget )
		self.xx = np.linspace(xl,xu,Nx)
		return self.xx

tt = np.linspace(0.,0.1,5)
Ipars = integrator_pars(delta,scale,dxTarg,tt)

def integrate_step(xmode,Jf,integrator_pars,cond):
	integrated = False
	while not integrated:
		xx = integrator_pars.construct_xx(xmode)
		j0 = Jf(xx)
		jt = odeint(dJdt,j0,integrator_pars.tt,args=(xx,dx))[-1,]
		if cond(jt):
			integrated = True
		else:
			# Modify pars
			# Expand delta
			integrator_pars.delta *= integrator_pars.scale


dxTarg = 0.001
def J0f(x):
	return 100*(x-x0)**2
xmode = 0.3

delta = [0.25,0.5]
scale = 1.1
dxTarg = 0.05

xl = xmode-scale*delta[0]
xu = xmode+scale*delta[1]
Nx = np.ceil( (xu-xl)/dxTarg )
xx = np.linspace(xl,xu,Nx)
j0 = J0f(xx)
jt = odeint(dJdt,j0,tt,args=(xx,xx[1]-xx[0]))[-1,]


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xx,jt)
ax.set_xlim((xl-0.1,xu+0.1))
ax.plot([xl,xl],[0,jt.max()],'k-.')
ax.plot([xu,xu],[0,jt.max()],'k-.')
ax.set_ylim((0,jt.max()))
ax.set_title("t=0.1, n=1")

zL = np.exp(-2*jt[0])
zU = np.exp(-2*jt[-1])

print Nx,zL,zU

xl -= 0.25
xu += 0.15

Nx = np.ceil( (xu-xl)/dxTarg )
xx = np.linspace(xl,xu,Nx)
j0 = J0f(xx)
jt = odeint(dJdt,j0,tt,args=(xx,xx[1]-xx[0]))[-1,]

fig2 = plt.figure()
ax=fig2.add_subplot(111)
ax.plot(xx,jt)
ax.set_xlim((xl-0.1,xu+0.1))
ax.plot([xl,xl],[0,jt.max()],'k-.')
ax.plot([xu,xu],[0,jt.max()],'k-.')
ax.set_ylim((0.,jt.max()))
ax.set_title("t=0.1, n=2")

zL = np.exp(-2*jt[0])
zU = np.exp(-2*jt[-1])

print Nx,zL,zU

f = myExtrap(xx,jt,[xx.min(),xx.max()])

xl -= 0.50
xu += 0.25
Nx = np.ceil( (xu-xl)/dxTarg )
xx = np.linspace(xl,xu,Nx)
dx = xx[1]-xx[0]


j0 = np.array([f(x) for x in xx])
j0 += abs(j0.min())

ax.plot(xx,j0)

jt = odeint(dJdt,j0,tt,args=(xx,dx))[-1,]

fig3 = plt.figure()
ax = fig3.add_subplot(111)
ax.plot(xx,jt)
ax.set_title("t=0.2,n=1")
ax.set_ylim((0.,jt.max()))


zL = np.exp(-2*jt[0])
zU = np.exp(-2*jt[-1])

print Nx,zL,zU

#############

f = myExtrap(xx,jt,[xx.min(),xx.max()])
xl -= 0.35
xu += 0.25
Nx = np.ceil( (xu-xl)/dxTarg )
xx = np.linspace(xl,xu,Nx)
dx = xx[1]-xx[0]

j0 = np.array([f(x) for x in xx])
j0 += abs(j0.min())
ax.plot(xx,j0)


jt = odeint(dJdt,j0,tt,args=(xx,dx))[-1,]

fig4 = plt.figure()
ax = fig4.add_subplot(111)
ax.plot(xx,jt)
ax.set_title("t=0.3,n=1")
ax.set_ylim((0.,jt.max()))

zL = np.exp(-2*jt[0])
zU = np.exp(-2*jt[-1])

print Nx,zL,zU

#############

f = myExtrap(xx,jt,[xx.min(),xx.max()])
xl -= 0.1
xu += 0.
Nx = np.ceil( (xu-xl)/dxTarg )
xx = np.linspace(xl,xu,Nx)
dx = xx[1]-xx[0]

j0 = np.array([f(x) for x in xx])
j0 += abs(j0.min())
ax.plot(xx,j0)

jt = odeint(dJdt,j0,tt,args=(xx,dx))[-1,]

fig5 = plt.figure()
ax = fig5.add_subplot(111)
ax.plot(xx,jt)
ax.set_title("t=0.4,n=1")
ax.set_ylim((0.,jt.max()))

zL = np.exp(-2*jt[0])
zU = np.exp(-2*jt[-1])

print Nx,zL,zU

#############

f = myExtrap(xx,jt,[xx.min(),xx.max()])
xl -= 0.1
xu += 0.
Nx = np.ceil( (xu-xl)/dxTarg )
xx = np.linspace(xl,xu,Nx)
dx = xx[1]-xx[0]

j0 = np.array([f(x) for x in xx])
j0 += abs(j0.min())
ax.plot(xx,j0)

jt = odeint(dJdt,j0,tt,args=(xx,dx))[-1,]

fig5 = plt.figure()
ax = fig5.add_subplot(111)
ax.plot(xx,jt)
ax.set_title("t=0.5,n=1")
ax.set_ylim((0.,jt.max()))

zL = np.exp(-2*jt[0])
zU = np.exp(-2*jt[-1])

print Nx,zL,zU

plt.show()

