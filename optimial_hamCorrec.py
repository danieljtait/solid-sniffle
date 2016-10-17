import numpy as np 
import matplotlib.pyplot as plt 
from main import init_dbw_example,potential,Hamiltonian,pStationary,HJacobi_integrator
from main import pTransition, pTransition2
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

x0 = 0.4
T = 0.5
tt = np.linspace(0.,T,100)
def J0f(x):
	return 100*(x-x0)**2

H = Hamiltonian(U,lambda x: 0.5)
Pst = pStationary(H)
dX = Diffusion(H)

NSim = 100
n = 500
rSample = np.zeros(NSim)

# R
# - Costly, used for plotting purposes
R = np.zeros((NSim,n))

# 35878
seed = np.random.choice(range(100000))
print seed

np.random.seed(37011)
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(NSim):
	r = dX.sim(x0,T,n)
	#if i % NSim/5 == 0 :
	#	ax.plot(np.linspace(0.,T,n),r,'k-')
	#rSample[i] = r[-1]
	R[i,] = r
ax.plot(T*np.ones(NSim),R[:,-1],'ro')
ax.set_xlim((0.,0.6))

def ObjFunc(par):
	# Create a new add_term for the Hamiltonian object
	def func(x,p):
		return par*(p-H.seperatrix(x))**2
	H.set_add_term(func)
	# Make transition PDF
	try :
		# Do something
		H.set_add_term(func)
		J = HJacobi_integrator(H,Pst)
		pT_1 = pTransition(J)
		pT_1.make(x0,tt)
		l = np.log(pT_1(R[:,-1]))
		return -np.sum(l)
	except:
		# Do something else
		return np.inf

#for e in np.linspace(-1.,1.,10) :
#	print ObjFunc(e)

from scipy.optimize import minimize
res = minimize(ObjFunc,0.1,method='Nelder-Mead')
print res 

#class ares:
#	def __init__(self):
#		self.x = 0.07539063
#res = ares()

def func(x,p):
	return res.x*(p-H.seperatrix(x))**2
H.set_add_term(func)
J = HJacobi_integrator(H,Pst)
pT = pTransition(J)
pT.make(x0,tt)

xx = np.linspace(-1.5,1.5,500)
fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.plot(xx,pT(xx))

from sklearn import mixture
clf = mixture.GaussianMixture(n_components=3,covariance_type='full')
clf.fit(R[:,-1].reshape(NSim,1))


p = np.zeros(xx.size)

print clf.means_
print clf.covariances_
#for i,w in enumerate(clf.weights_):
#	p += w*norm.pdf(xx,clf.means_[i][0],clf.covariances_[i][0,0])
#ax.plot(xx,p,'r-')

q = pT(xx)
fig3 = plt.figure()
ax = fig3.add_subplot(111)
scale = 0.1
ax.plot(scale*(q) + T,xx,'k-')
ax.plot(T*np.ones(NSim),R[:,-1],'r+')

tt = np.linspace(0.,T,n)
ax.plot(tt,R[18,],'k-')
ax.plot(tt,R[1,],'k-')
ax.plot(tt,R[9,],'k-')

minInd = np.where(R[:,-1] == R[:,-1].min())
maxInd = np.where(R[:,-1] == R[:,-1].max())

print minInd[0][0],maxInd[0][0]

ax.plot(tt,R[minInd[0][0],],'k-')
ax.plot(tt,R[maxInd[0][0],],'k-')


ax.set_ylim((-1.5,1.5))
ax.set_xlim((0,T+scale*q.max()+0.1))

ax.set_xlabel("Time")
ax.set_ylabel("X_T")

plt.show()