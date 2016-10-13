
import numpy as np 
import matplotlib.pyplot as plt 
from main import init_dbw_example,potential,Hamiltonian,pStationary,HJacobi_integrator
from main import pTransition
from scipy.interpolate import UnivariateSpline

from scipy.stats import norm
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

f,fg = init_dbw_example()
U = potential(f,fg)

H = Hamiltonian(U,lambda x: 1.)
Pst = pStationary(H)
dX = Diffusion(H)

NSim = 100
T = 0.5
rSample = np.zeros(NSim)

x0 = 0.1

tt = [0.]
rSample[0] = x0

t = 0.

seed = np.random.choice(range(100))
print seed
seed = 93
np.random.seed(seed)

fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(NSim-1) :
	r = dX.sim(rSample[i],T,50)


	ax.plot(np.linspace(t,t+T,50),r,'k-')

	t += T 

	tt.append(t)
	rSample[i+1] = r[-1]

ax.plot(tt,rSample,'o')

np.savetxt('dbwData1.txt',rSample,delimiter=",")

plt.show()
