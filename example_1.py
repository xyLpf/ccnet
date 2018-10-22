import numpy as np
from model import CCNet

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel,ConstantKernel
def data_fn(x):
	k=10.0
	return np.concatenate((np.sin(2*np.pi*k*x[:,[0]]),np.cos(2*np.pi*k*x[:,[0]])),1)
	
def get_data_fn(n_samples=1000,noise=0.2,seed=0):
	np.random.seed(seed)
	X=np.random.random(size=(n_samples,1))*2-1
	Y=data_fn(X)+np.random.normal(0.0,np.ones((X.shape[0],2))*noise)
	return X,Y
	

np.random.seed(0)

#n_epochs=500
n_epochs=1000
batch_size=700


X,Y = get_data_fn(n_samples=700,noise=0.0,seed=0)
kernel=1.0*RBF(0.01,[0.001,10.0])+WhiteKernel()
gp=GaussianProcessRegressor(kernel)
gp.fit(X,Y)

Xp=np.linspace(-1,1,500).reshape(-1,1)

yp_gp=gp.predict(Xp,return_std=False)



cc=CCNet(50,do_rate=0.3,id_dropout=0.0)
cc.fit(X,Y,epochs=n_epochs,batch_size=batch_size,verbose=True)


Yp2=cc.predict(Xp)

Yp,ypstd=cc.stochastic_predict(Xp,100)


y_real=data_fn(Xp)

print("Stochastic: " + str(np.mean(np.sum((y_real-Yp)**2,1))))
print("Det. : " + str(np.mean(np.sum((y_real-Yp2)**2,1))))
print("GP: " + str(np.mean(np.sum((yp_gp-y_real)**2,1))))
import matplotlib.pyplot as plt
plt.plot(X,Y,'r.')
plt.plot(Xp,Yp)
plt.plot(Xp,yp_gp)
plt.plot(Xp,y_real)
#plt.plot(Xp,Yp2)
plt.fill_between(Xp[:,0],Yp[:,0]-2*ypstd[:,0],Yp[:,0]+2*ypstd[:,0],alpha=0.5)
plt.show()

		

