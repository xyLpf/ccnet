import numpy as np
from get_data import get_data_fn,data_fn
from model import CCNet

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel,ConstantKernel



def f(x):
	return (x+0.5>=0)*np.sin(64*(x+0.5)**4)#-1.0*(x>0)+numpy.

np.random.seed(0)
x_train=np.random.random(size=(70,1))-0.5
y_train=f(x_train)+np.random.normal(0.0,0.01,size=x_train.shape)

#n_epochs=500
n_epochs=2000
batch_size=700


X,Y = get_data_fn(n_samples=700,noise=0.0,seed=0)
kernel=1.0*RBF(0.01,[0.001,10.0])+WhiteKernel()
gp=GaussianProcessRegressor(kernel)
gp.fit(X,Y)

Xp=np.linspace(-1,1,500).reshape(-1,1)

yp_gp=gp.predict(Xp,return_std=False)



cc=CCNet(100,do_rate=0.2,id_dropout=0.0)
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

		

