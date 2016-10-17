
import numpy as np 
import matplotlib.pyplot as plt 
from main import init_dbw_example,potential,Hamiltonian,pStationary,HJacobi_integrator
from main import pTransition,pTransition2
from scipy.interpolate import UnivariateSpline
"""
Investigating the addition of terms to the Hamiltonian
"""

f,fg = init_dbw_example()
U = potential(f,fg)

x0 = 0.4
T = .5
tt = np.linspace(0.,T,100)
def J0f(x):
	return 100*(x-x0)**2

H = Hamiltonian(U,lambda x: 0.5)
Pst = pStationary(H)
J = HJacobi_integrator(H,Pst)
Jt_1 = J(tt,J0f)

from JIntegrator import integrator_pars


"""
"""
xx = J.xx
fig = plt.figure()
ax = fig.add_subplot(111)

pT_1 = pTransition(J)
pT_1.make(x0,tt)

intPar = integrator_pars(delta=[0.3,0.3],scale=1.25,dxTarget=0.005)
pT_2 = pTransition2(H,intPar)
pT_2.make(x0,T)

#ax.plot(xx,pT_1(xx),'b-')
ax.plot(xx,pT_2(xx))

for eps in [0.03,0.04,0.05,0.06] :
	def func(x,p):
		return eps*(p-H.seperatrix(x))**2
	print eps
	try:
		H.set_add_term(func)
		J = HJacobi_integrator(H,Pst)

#	pT_1 = pTransition(J)
#	pT_1.make(x0,tt)

	# Reset intPar
		intPar = integrator_pars(delta=[0.25,0.25],scale=1.25,dxTarget=0.00001)

		pT_2 = pTransition2(H,intPar)
		pT_2.make(x0,T)
#	ax.plot(xx,pT_1(xx))
		ax.plot(xx,pT_2(xx))
	except:
		print "Fail"
plt.show()

