
import numpy as np 
import matplotlib.pyplot as plt 
from main import init_dbw_example,potential,Hamiltonian,pStationary,HJacobi_integrator
from main import pTransition
from scipy.interpolate import UnivariateSpline
"""
Investigating the addition of terms to the Hamiltonian
"""

f,fg = init_dbw_example()
U = potential(f,fg)

x0 = 0.4
T = 0.5
tt = np.linspace(0.,T,100)
def J0f(x):
	return 100*(x-x0)**2

H = Hamiltonian(U,lambda x: 0.5)
Pst = pStationary(H)
J = HJacobi_integrator(H,Pst)
Jt_1 = J(tt,J0f)

"""
"""
xx = J.xx
fig = plt.figure()
ax = fig.add_subplot(111)
pT_1 = pTransition(J)
pT_1.make(x0,tt)
ax.plot(xx,pT_1(xx))

for eps in [0.05,0.1,0.2] :
	def func(x,p):
		return eps*(p-H.seperatrix(x))**2
	print eps
	H.set_add_term(func)
	J = HJacobi_integrator(H,Pst)

	pT_1 = pTransition(J)
	pT_1.make(x0,tt)
	ax.plot(xx,pT_1(xx))

	#Jt_2 = J(tt,J0f)
	#ax.plot(xx,np.exp(-Jt_2))

plt.show()

