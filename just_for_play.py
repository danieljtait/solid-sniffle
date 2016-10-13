
import numpy as np 
import matplotlib.pyplot as plt 



def D(x,xlim) :
	if x < xlim[0] :
		return 0.
	else :
		return np.exp(-0.1*(x-xlim[0])**2)



xx = np.linspace(0.,10,100)

fig = plt.figure()
ax = fig.add_subplot(111)

x = [1.,1.,2.5,3.1,3.1]
y = [0.,1.,1.,0.8,0.]

ax.plot(x,y)
ax.plot(xx,np.zeros(xx.size),'-k')

ax.set_ylim((0,1.5))

plt.show()