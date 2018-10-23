import keras
from keras.models import Model
from keras.layers import SimpleRNN,Dense,Multiply,Activation,Input,Add,Subtract,Reshape,Concatenate,Lambda,GRU,Flatten,GaussianNoise,Dropout,LSTM,TimeDistributed,Bidirectional,ActivityRegularization,RepeatVector
from sklearn.neighbors import KDTree
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import keras.regularizers

def lamb1(x):
	return x[:,:,0:-no]
def lamb1_os(x):
	ll=list(x)
	ll[-1]=ll[-1]-no
	return tuple(ll)

def lamb2(x):
	sh=tf.shape(x)
	#sh[-1]=1
	return x[:,:,-no::]
def lamb2_os(x):
	ll=list(x)
	ll[-1]=no
	return tuple(ll)

def lamb3(x):
	return tf.reduce_sum(x,1)
def lamb3_os(x):
	ll=list(x)
	ll=[ll[0],ll[-1]]
	return tuple(ll)

	
	
def get_model2(n_neighbors,n_dim,n_out,alpha=1e-4):
	
	
	
	inp1=Input((n_dim,))
	inp2=Input((n_neighbors,n_dim+n_out))
	inp_mask=Input((n_neighbors,1))
	
	l1 = Lambda(lamb1,output_shape=lamb1_os)
	l2 = Lambda(lamb2,output_shape=lamb2_os)
	l3 = Lambda(lamb3,output_shape=lamb3_os)
	
	h1=inp1
	h=Concatenate()([RepeatVector(n_neighbors)(h1),inp2])
	h=Multiply()([h,inp_mask])
	h=Concatenate()([h,inp_mask])
	h=LSTM(32,activation='relu',return_sequences=True)(h)
	h=LSTM(n_out,return_sequences=False)(h)
	out=h#Dense(n_out)(h)
	
	model=Model([inp1,inp2,inp_mask],out)
	model.compile(loss='mse',optimizer=Adam(1e-3))
	return model
def get_model(n_neighbors,n_dim=1,n_out=1,alpha=1e-10):
	l1 = Lambda(lamb1,output_shape=lamb1_os)
	l2 = Lambda(lamb2,output_shape=lamb2_os)
	l3 = Lambda(lamb3,output_shape=lamb3_os)
	inp1 = Input((n_dim,))
	inp2 = Input((n_neighbors,n_dim+n_out))
	inp_mask = Input((n_neighbors,1))
	
	sc10 = Lambda(lambda x: 10.0*x)
	
	h=l1(inp2)
	hi=RepeatVector(n_neighbors)(inp1)
	noise=Lambda(lambda x: tf.random_normal(tf.shape(x),0.0,0.1))(inp2)
	#noise=RepeatVector(n_neighbors)(noise)
	
	h=Add()([h,noise])
	hi=Add()([hi,noise])
	
	
	
	d_0=TimeDistributed(Dense(32,activation='relu',kernel_regularizer=keras.regularizers.l2(alpha)))
	d_02=TimeDistributed(Dense(1,kernel_regularizer=keras.regularizers.l2(alpha)))
	
	
	hh = d_02(d_0(h))
	it = d_02(d_0(hi))
	
	h=Subtract()([hh,it])
	
	h2=sc10(h)
	h=Concatenate()([h,h2])
	h=Multiply()([h,inp_mask])
	h=Concatenate()([h,inp_mask])
	
	d_1=Dense(64,activation='relu',kernel_regularizer=keras.regularizers.l2(alpha))
	d_2=Dense(64,activation='relu',kernel_regularizer=keras.regularizers.l2(alpha))
	d_3=Dense(n_out,kernel_regularizer=keras.regularizers.l2(alpha))
	
	h=TimeDistributed(d_1)(h)
	h=TimeDistributed(d_2)(h)
	h=TimeDistributed(d_3)(h)

	h=Lambda(lambda x:tf.nn.softmax(x,1))(h)

	h=Multiply()([h,l2(inp2)])
	print(h)
	h = Lambda(lambda x:tf.reduce_sum(x,1))(h)

	out=h#Dense(n_out)(h)
	
	
	
	model=Model([inp1,inp2,inp_mask],out)
	model.compile(loss='mse',optimizer=Adam(1e-3))
	return model
	
	
	
	
class CCNet:
	def __init__(self,n_neighbours,do_rate=0.2,id_dropout=0.9,shuffle=False,model=None):
		self.nn=n_neighbours
		self.do_rate=do_rate
		self.id_dropout=id_dropout
		self.shuffle=shuffle
		self.inp_model=model
	def fit(self,X,Y,epochs=10,batch_size=128,verbose=True):
		self.n_dim=X.shape[1]
		self.n_out=Y.shape[1]
		
		global no
		no=self.n_out
		
		if self.inp_model is None:	
		
			self.model=get_model(self.nn,self.n_dim,self.n_out)
		else:
			self.model = self.inp_model
		
		self.kdt=KDTree(X)
		
		m=int(X.shape[0]/batch_size)
		
		self.X=np.copy(X)
		self.Y=np.copy(Y)
		
		
		X2=np.zeros((X.shape[0],self.nn,self.n_dim+self.n_out))
		
		self.X2=np.copy(X2)
		
		idx=self.kdt.query(X,self.nn+1,return_distance=False)
		idx=idx[:,0:-1]
		
		for i in range(0,X.shape[0]):
			for j in range(0,self.n_dim):
				X2[i,:,j]=X[idx[i,:],j]
			for j in range(0,self.n_out):
				X2[i,:,self.n_dim+j]=self.Y[idx[i,:],j]
		print(X.shape,X2.shape)
		
		
		outp_noise=0.0
		inp_noise=0.0
		for i in range(0,epochs):
			
			for j in range(0,m):
				if self.shuffle:
					idx2=np.transpose(np.copy(idx))
					np.random.shuffle(idx2)
					idx=np.copy(idx2).T
					for k in range(0,X.shape[0]):
						for l in range(0,self.n_dim):
							X2[k,:,l]=X[idx[k,:],l]
						for l in range(0,self.n_out):
							X2[k,:,self.n_dim+l]=self.Y[idx[k,:],l]
							
				batch_x1 = X[j*batch_size:(j+1)*batch_size]
				batch_x2 = X2[j*batch_size:(j+1)*batch_size]
				mask = np.random.random(size=(batch_x2.shape[0],batch_x2.shape[1]))>=self.do_rate
				mask[:,0] = mask[:,0]*(np.random.random(size=(batch_x2.shape[0],))>=self.id_dropout)
				#print(np.all(mask))
				batch_mask=mask.reshape(mask.shape[0],mask.shape[1],1)#/(1.0-self.do_rate)
				#batch_mask
				
				batch_y=Y[j*batch_size:(j+1)*batch_size]+np.random.normal(0.0,outp_noise,size=(batch_size,self.n_out))

				#print(batch_y.shape,batch_x1.shape,batch_x2.shape,batch_mask.shape)
				#idx = self.kdt.query(batch_x1,self.nn,return_distance=False)
				ll=self.model.train_on_batch([batch_x1,batch_x2,batch_mask],batch_y)
				if verbose:
					print("Epoch: "+ str(i)+". Loss: "+str(ll))
			if X.shape[0] > m*batch_size:
				if self.shuffle:
						idx2=np.transpose(np.copy(idx))
						np.random.shuffle(idx2)
						idx=np.copy(idx2).T
						for k in range(0,X.shape[0]):
							for l in range(0,self.n_dim):
								X2[k,:,l]=X[idx[k,:],l]
							for l in range(0,self.n_out):
								X2[k,:,self.n_dim+l]=self.Y[idx[k,:],l]
				bs=X.shape[0]-m*batch_size
				batch_x1 = X[m*batch_size::]
				batch_x2 = X2[m*batch_size::]
				mask = np.random.random(size=(bs,batch_x2.shape[1]))>=self.do_rate
				mask[:,0] = mask[:,0]*(np.random.random(size=(bs,))>=self.id_dropout)
				#print(np.all(mask))
				batch_mask=mask.reshape(mask.shape[0],mask.shape[1],1)#/(1.0-self.do_rate)
				#batch_mask
				
				batch_y=Y[m*batch_size::]+np.random.normal(0.0,outp_noise,size=(bs,self.n_out))

				#print(batch_y.shape,batch_x1.shape,batch_x2.shape,batch_mask.shape)
				#idx = self.kdt.query(batch_x1,self.nn,return_distance=False)
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
				X2[i,:,j]=self.X[idx[i,:],j]
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
				X2[i,:,j]=self.X[idx[i,:],j]
			for j in range(0,self.n_out):
				X2[i,:,self.n_dim+j]=self.Y[idx[i,:],j]
			
			
		preds = np.zeros((n_iter,X.shape[0],self.n_out))
		
		for n in range(0,n_iter):
			if self.shuffle:
				idx2=np.transpose(np.copy(idx))
				np.random.shuffle(idx2)
				idx=np.copy(idx2).T
			X2=np.zeros((X.shape[0],self.nn,self.n_dim+self.n_out))
			for i in range(0,X2.shape[0]):
				for j in range(0,self.n_dim):
					X2[i,:,j]=self.X[idx[i,:],j]
				for j in range(0,self.n_out):
					X2[i,:,self.n_dim+j]=self.Y[idx[i,:],j]	
			mask = np.random.random(size=(X.shape[0],self.nn))>=self.do_rate
			#mask[:,0] = mask[:,0]*(np.random.random(size=(X.shape[0],))>=self.id_dropout)
			mask=mask.reshape(mask.shape[0],mask.shape[1],1)#/(1.0-self.do_rate)
			#X2p=np.zeros_like(X2)
			#for k in range(0,X.shape[0]):
			#	kk = int(np.sum(mask[k]))
			#	X2p[k,0:kk]=X2[k,mask[k]]
			preds[n]=self.model.predict([X,X2,mask])
		
		return  np.mean(preds,0),np.std(preds,0)
		