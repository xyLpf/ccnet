import keras
from keras.models import Model
from keras.layers import Dense,Multiply,Activation,Input,Add,Subtract,Reshape,Concatenate,Lambda,GRU,Flatten,GaussianNoise,Dropout,LSTM,TimeDistributed,Bidirectional,ActivityRegularization,RepeatVector
from sklearn.neighbors import KDTree
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import keras.regularizers

def lamb1(x):
	return x[:,:,0:-1]
def lamb1_os(x):
	ll=list(x)
	ll[-1]=ll[-1]-1
	return tuple(ll)

def lamb2(x):
	sh=tf.shape(x)
	#sh[-1]=1
	return tf.reshape(x[:,:,-1],[sh[0],sh[1],1])
def lamb2_os(x):
	ll=list(x)
	ll[-1]=1
	return tuple(ll)
def lamb3(x):
	return tf.reduce_sum(x,1)
def lamb3_os(x):
	ll=list(x)
	ll=[ll[0],ll[-1]]
	return tuple(ll)
		
def lamb3s(x):
	return tf.nn.moments(x,1)[1]
def lamb3s_os(x):
	ll=list(x)
	ll=[ll[0],ll[-1]]
	return tuple(ll)
	
def lamb4(x):
	sh=tf.shape(x)
	return tf.reshape(tf.reduce_mean(x**2,-1),[sh[0],sh[1],1])
def lamb4_os(x):
	ll=list(x)
	ll[-1]=1
	return tuple(ll)
def lamb5(x):
	return tf.exp(-x)
def lamb6(x):
	sh=tf.shape(x)
	return x/(1e-7+tf.reshape(tf.reduce_sum(x,1),[sh[0],1,sh[2]]))

def lamb7(x):
	sh=tf.shape(x)
	return tf.reshape(x[:,0,:],[sh[0],1,sh[2]])
def lamb7_os(x):
	ll=list(x)
	ll[1]=1
	return tuple(ll)
	
def lamb8(x):
	sh=tf.shape(x)
	return tf.reshape(x[:,1::,:],[sh[0],sh[1]-1,sh[2]])
def lamb8_os(x):
	ll=list(x)
	ll[1]=ll[1]-1
	return tuple(ll)

	
def get_model3(n_neighbors,n_dim=1,n_out=1):
	l1 = Lambda(lamb1,output_shape=lamb1_os)
	l2 = Lambda(lamb2,output_shape=lamb2_os)
	l3 = Lambda(lamb3,output_shape=lamb3_os)
	l3s = Lambda(lamb3s,output_shape=lamb3s_os)
	inp1 = Input((n_dim,))
	inp2 = Input((n_neighbors,n_dim+n_out))
	inp_mask = Input((n_neighbors,1))
	
	
	#h=l1(inp2)
	#h=GaussianNoise(0.01)(h)
	h=inp2
	h=Multiply()([h,inp_mask])
	h=Concatenate()([h,inp_mask])
	
	#h=Bidirectional(LSTM(16,activation='relu',return_sequences=True))(h)
	h=Bidirectional(LSTM(4,return_sequences=False))(h)
	#h=Dense(64,activation='relu')(h)
	h=Dense(n_out)(h)
	out=h
	model=Model([inp1,inp2,inp_mask],out)
	return model	

def get_model4(n_neighbors,n_dim=1,n_out=1):
	l1 = Lambda(lamb1,output_shape=lamb1_os)
	l2 = Lambda(lamb2,output_shape=lamb2_os)
	l3 = Lambda(lamb3,output_shape=lamb3_os)
	l3s = Lambda(lamb3s,output_shape=lamb3s_os)
	inp1 = Input((n_dim,))
	inp2 = Input((n_neighbors,n_dim+n_out))
	inp_mask = Input((n_neighbors,1))
	
	sc10 = Lambda(lambda x: 10.0*x)
	
	h=l1(inp2)
	h2=sc10(h)
	h3=sc10(h2)
	h4=sc10(h3)
	h=Concatenate()([h,h2,h3,h4])
	h=Multiply()([h,inp_mask])
	h=Concatenate()([h,inp_mask])
	
	
	d_1=Dense(32,activation='tanh',kernel_regularizer=keras.regularizers.l2(1e-9))
	d_2=Dense(32,activation='tanh',kernel_regularizer=keras.regularizers.l2(1e-9))
	d_3=Dense(1)
	
	h=TimeDistributed(d_1)(h)
	h=TimeDistributed(d_2)(h)
	h=TimeDistributed(d_3)(h)
	#h=TimeDistributed(Dense(1,use_bias=False))(h)
	h=Lambda(lambda x:tf.nn.softmax(x,1))(h)
	#h=Lambda(lambda x:x/tf.reshape(tf.reduce_sum(x,1),[tf.shape(x)[0],1,1]))(h)
	h=Multiply()([h,l2(inp2)])
	h = l3(h)
	#h = l3s(h)
	#h=Concatenate()([h1,h2])
	#h=Dense(64,activation='relu')(h)
	out=h#Dense(n_out)(h)
	
	
	
	model=Model([inp1,inp2,inp_mask],out)
	
	cc=Concatenate()([inp1,Lambda(lambda x: tf.ones([tf.shape(x)[0],1]))(inp1)])
	cc2=Concatenate()([inp1,Lambda(lambda x: tf.zeros([tf.shape(x)[0],1]))(inp1)])
	#gg=d_3(d_2(d_1(cc)))
	#gg2=d_3(d_2(d_1(cc2)))
	#model_trans=Model(inp1,gg)
	#model_trans2=Model(inp1,gg2)
	return model,0,0#model_trans,model_trans2
	
	
	
	
class CCNet:
	def __init__(self,n_neighbours,do_rate=0.2,id_dropout=0.9):
		self.nn=n_neighbours
		self.do_rate=do_rate
		self.id_dropout=id_dropout
	def fit(self,X,Y,epochs=10,batch_size=128,verbose=True):
		self.n_dim=X.shape[1]
		self.n_out=Y.shape[1]
		self.model,self.model_trans,self.model_trans2=get_model4(self.nn,self.n_dim,self.n_out)
		self.model.compile(loss='mse',optimizer=Adam(1e-3))
		self.kdt=KDTree(X)
		
		m=int(X.shape[0]/batch_size)
		
		self.X=np.copy(X)
		self.Y=np.copy(Y)
		
		
		X2=np.zeros((X.shape[0],self.nn,self.n_dim+self.n_out))
		
		self.X2=np.copy(X2)
		
		idx=self.kdt.query(X,self.nn+1,return_distance=False)
		idx=idx[:,1::]
		
		for i in range(0,X.shape[0]):
			for j in range(0,self.n_dim):
				X2[i,:,j]=(X[idx[i,:],j]-X[i,j])
			for j in range(0,self.n_out):
				X2[i,:,self.n_dim+j]=self.Y[idx[i,:],j]
		print(X.shape,X2.shape)
		
		
		outp_noise=0.0
		inp_noise=0.0
		for i in range(0,epochs):
			for j in range(0,m):
				batch_x1 = X[j*batch_size:(j+1)*batch_size]
				batch_x2 = X2[j*batch_size:(j+1)*batch_size]
				mask = np.random.random(size=(batch_x2.shape[0],batch_x2.shape[1]))>=self.do_rate
				#mask[:,0] = np.random.random(size=(batch_x2.shape[0],))>=self.id_dropout
				#print(np.all(mask))
				batch_mask=mask.reshape(mask.shape[0],mask.shape[1],1)#/(1.0-self.do_rate)
				
				
				batch_y=Y[j*batch_size:(j+1)*batch_size]+np.random.normal(0.0,outp_noise,size=(batch_size,self.n_out))

				
				idx = self.kdt.query(batch_x1,self.nn,return_distance=False)
				ll=self.model.train_on_batch([batch_x1,batch_x2,batch_mask],batch_y)
				if verbose:
					print("Epoch: "+ str(i)+". Loss: "+str(ll))
	def predict(self,X):
		assert(X.shape[1]==self.n_dim)
		
		
		idx = self.kdt.query(X,self.nn,return_distance=False)
		X2=np.zeros((X.shape[0],self.nn,self.n_dim+self.n_out))
		mask=np.ones((X2.shape[0],self.nn,1))
		for i in range(0,X2.shape[0]):
			for j in range(0,self.n_dim):
				X2[i,:,j]=(self.X[idx[i,:],j]-X[i,j])
			for j in range(0,self.n_out):
				X2[i,:,self.n_dim+j]=self.Y[idx[i,:],j]
		#print(X2)
		return self.model.predict([X,X2,mask])
		
	def stochastic_predict(self,X,n_iter=100):
		assert(X.shape[1]==self.n_dim)
		
		idx = self.kdt.query(X,self.nn,return_distance=False)
		#if np.all(X==self.X):
		#idx=idx[:,1::]
		#else:
		#	idx=idx[:,0:-1]
			
		#idx = self.kdt.query(X,self.nn,return_distance=False)
		X2=np.zeros((X.shape[0],self.nn,self.n_dim+self.n_out))
		for i in range(0,X2.shape[0]):
			for j in range(0,self.n_dim):
				X2[i,:,j]=self.X[idx[i,:],j]-X[i,j]
			for j in range(0,self.n_out):
				X2[i,:,self.n_dim+j]=self.Y[idx[i,:],j]
			
			
		preds = np.zeros((n_iter,X.shape[0],1))
		
		for i in range(0,n_iter):
			mask = np.random.random(size=(X.shape[0],self.nn))>=self.do_rate
			mask=mask.reshape(mask.shape[0],mask.shape[1],1)#/(1.0-self.do_rate)
			#X2p=np.zeros_like(X2)
			#for k in range(0,X.shape[0]):
			#	kk = int(np.sum(mask[k]))
			#	X2p[k,0:kk]=X2[k,mask[k]]
			preds[i]=self.model.predict([X,X2,mask])
		
		return  np.mean(preds,0),np.std(preds,0)
		