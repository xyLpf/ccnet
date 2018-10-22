import numpy as np
from model import CCNet

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel,ConstantKernel,Matern



def f(x):
	return (x+0.5>=0)*np.sin(64*(x+0.5)**4)#-1.0*(x>0)+numpy.

np.random.seed(0)
X=np.random.random(size=(70,1))-0.5
Y=f(X)+np.random.normal(0.0,0.01,size=X.shape)
#n_epochs=500
n_epochs=5000
batch_size=70

kernel=1.0*Matern(0.01,[0.001,10.0],nu=1.5)+WhiteKernel()
gp=GaussianProcessRegressor(kernel)
gp.fit(X,Y)

Xp=np.linspace(-0.5,0.5,500).reshape(-1,1)

yp_gp=gp.predict(Xp,return_std=False)



cc=CCNet(69,do_rate=0.1,id_dropout=0.0)
cc.fit(X,Y,epochs=n_epochs,batch_size=batch_size,verbose=True)
Yp2=cc.predict(Xp)
ypx,_=cc.stochastic_predict(X,100)#predict(X)
print(np.mean(np.sum((Y-ypx)**2,1)))


Yp,ypstd=cc.stochastic_predict(Xp,100)


y_real=f(Xp)

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

#Xp=np.linspace(-0.5,0.5,5000).reshape(-1,1)
#plt.figure()
#plt.plot(Xp,cc.model_trans.predict(Xp))
#plt.plot(Xp,cc.model_trans2.predict(Xp))
plt.show()

		

