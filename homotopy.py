
import numpy as np 
import matplotlib.pyplot as plt 
from main import init_dbw_example,potential,Hamiltonian,pStationary,HJacobi_integrator
from scipy.interpolate import UnivariateSpline
"""
Investigating properties of the homotopy method
"""

f,fg = init_dbw_example()
U = potential(f,fg)
H = Hamiltonian(U,lambda x: 0.5)
Pst = pStationary(H)
J = HJacobi_integrator(H,Pst)


x0 = 0.3
T = 0.3
tt = np.linspace(0.,T,100)

Jt = J(tt,lambda x : 10*(x-x0)**2)
xx = J.xx
dx = J.dx

"""
Plot the gradient over the contours of the Hamiltonian
"""
fig = plt.figure()
ax = fig.add_subplot(111)

# Smooth the oscillations 
dJdx = np.gradient(Jt,dx)
#xx = xx[abs(dJdx) < 5.]
#dJdx = dJdx[abs(dJdx) < 5.]


ax.plot(xx,dJdx,'b')
ax.plot(xx,H.seperatrix(xx),'r')

s = H.seperatrix(xx)

for w in np.linspace(1./3,2./3,3):

	ax.plot(xx,w*H.seperatrix(xx) + (1-w)*dJdx,'g')

pp = np.linspace(-5.,5.,xx.size)
X,P = np.meshgrid(xx,pp)
Z = H(X.ravel(),P.ravel())
Z = Z.reshape(X.shape)

cs = ax.contour(X,P,Z,colors='k')
ax.set_ylim((pp[0],pp[-1]))
ax.set_xlim((-1.,1.))
plt.show()