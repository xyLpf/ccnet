import numpy as np
from get_data import get_data_fn,data_fn
from model import CCNet

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel,ConstantKernel,Matern



def f(x):
	tmp=np.sin(2*np.pi*5*x[:,0])+np.cos(2*np.pi*x[:,1]*1.3)
	return tmp.reshape(-1,1)
	
np.random.seed(0)
X=np.random.random(size=(1000,2))-0.5
Y=f(X)+np.random.normal(0.0,0.1,size=(X.shape[0],1))




nx=100
ny=100

Xp=np.zeros((nx*ny,2))
for i in range(0,nx):
	for j in range(0,ny):
		Xp[i*ny+j,0]=-0.5 + float(i)/(nx-1)
		Xp[i*ny+j,1]=-0.5 + float(j)/(ny-1)


#n_epochs=500
n_epochs=2000
batch_size=1000

kernel=1.0*RBF([0.1,0.1],[[0.001,10.0],[0.001,10.0]])+WhiteKernel()
gp=GaussianProcessRegressor(kernel)
gp.fit(X,Y)

yp_gp=gp.predict(Xp,return_std=False)



cc=CCNet(20,do_rate=0.5,id_dropout=0.0)
print(X.shape,Y.shape)
cc.fit(X,Y,epochs=n_epochs,batch_size=batch_size,verbose=True)



Yp2=cc.predict(Xp)

Yp2,ypstd=cc.stochastic_predict(Xp,100)


y_real=f(Xp)

print("Det. : " + str(np.mean(np.sum((y_real-Yp2)**2,1))))
print("GP: " + str(np.mean(np.sum((yp_gp-y_real)**2,1))))


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xx=np.zeros((nx,ny))
yy=np.zeros((nx,ny))

z_real=np.zeros((nx,ny))
z_gp=np.zeros((nx,ny))
z_nn=np.zeros((nx,ny))

for i in range(0,nx):
	for j in range(0,ny):
		xx[i,j]=Xp[i*ny+j,0]
		yy[i,j]=Xp[i*ny+j,1]
		z_real[i,j]=y_real[i*ny+j,0]
		z_gp[i,j]=yp_gp[i*ny+j,0]
		z_nn[i,j]=Yp2[i*ny+j,0]




ax.plot_surface(xx,yy,z_real,alpha=0.5)
ax.plot_wireframe(xx,yy,z_nn,color='red')
plt.show()

