import numpy as np 
from main import init_dbw_example,potential,Hamiltonian,pStationary,HJacobi_integrator
from main import pTransition

class loglikelihood:
	def __init__(self,data,repData=None):
		self.data = data
		self.has_repr = False
		if repData != None :
			self.repData = repData
			self.has_repr = True

	def __call__(self,par,delT):
		self.tt = np.linspace(0.,delT,100)
		D2 = np.exp(par[1])
		def f(x):
			return -(-x**4 + 2*par[0]*x**2)
		def fgrad(x):
			return -(4*x*(par[0]-x**2))
		U = potential(f,fgrad)
		H = Hamiltonian(U,lambda x: D2)
		Pst = None
		if self.has_repr:
			return self.evalLL_wrep(H,Pst)
		else:
			return self.evalLL(H,Pst)
	def evalLL_wrep(self,HObj,Pst):
		try:
			val = 0.
			for i,R in zip(self.repData.inds,self.repData.Ireps):
				# Make Pt for i
				J  = HJacobi_integrator(HObj,Pst)
				pT = pTransition(J)
				pT.make(self.data[i],self.tt)
				for j in R :
					if j != (self.data.shape[0]-1):
						val += np.log(pT(self.data[j+1]))
			return -val
		except:
			return np.inf
	def evalLL(self,HObj,Pst) :
		try :
			val = 0.
			for i in range(self.data.shape[0]-1) :
				J  = HJacobi_integrator(HObj,Pst)
				pT = pTransition(J)
				pT.make(self.data[i],self.tt)
				val += -np.log(pT(self.data[i+1]))
			return val
		except:
			return np.inf

def test():
	from data_transform import rRepSample
	import matplotlib.pyplot as plt
	from scipy.optimize import minimize


	X = np.loadtxt('dbwData1.txt',delimiter=',')
	rep = rRepSample(X[:,None],0.09,11)
	rep.make()
	#rep = None
	T = 0.5
	logLik = loglikelihood(X,rep)

	res = minimize(logLik,[0.744152,np.log(0.5)],args=(T,))

	"""
	ll = []
	pars = np.linspace(0.6,0.9,3)
	for par in pars :
		val = logLik(par,T)
		ll.append(val)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(pars,ll)
	plt.show()
	"""
	print "--------------------------"
	print ""
	print res

test()