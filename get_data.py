import numpy as np
def data_fn(x):
	return np.sin(2*np.pi*10*x)
	
def get_data_fn(n_samples=1000,noise=0.2,seed=0):
	np.random.seed(seed)
	X=np.random.random(size=(n_samples,1))*2-1
	Y=data_fn(X)+np.random.normal(0.0,np.ones_like(X)*noise)
	return X,Y
	