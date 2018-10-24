import numpy as np
import pickle
import matplotlib.pyplot as plt
from model import CCNet
from keras.utils import to_categorical
from sklearn.neighbors import KDTree

train = pickle.load(open("data/train_mnist_emb.dat","rb"))
test = pickle.load(open("data/test_mnist_emb.dat","rb"))

print(train)

for i in range(0,10):
	idx=np.argwhere(test[1] == i)[:,0]
	
	plt.scatter(test[0][idx,0],test[0][idx,1])






x_train,y_train = train[0],to_categorical(train[1],10)
x_test,y_test = test[0],to_categorical(test[1],10)


xmu=np.mean(x_train,0)
xstd=np.std(x_train,0)

x_train=(x_train-xmu)/xstd
x_test=(x_test-xmu)/xstd

kdt=KDTree(x_train)
idx=kdt.query(x_test,1,return_distance=False)

y_pred=y_train[idx[:,0]]
print(np.mean(np.argmax(y_pred,1)==np.argmax(y_test,1)))

plt.show()


cc=CCNet(2000,do_rate=0.1,id_dropout=0.0,shuffle=False,loss='categorical_crossentropy')
cc.fit(x_train,y_train,epochs=15,batch_size=128,verbose=True)


#Yp2=cc.predict(x_test)
Yp2,ypstd=cc.stochastic_predict(x_test,50,verbose=True)

print(ypstd.shape)

err=np.sqrt(np.sum(ypstd**2,1))
ass = np.argsort(err)
print(ass.shape)
aa=1*(np.argmax(y_test,1)==np.argmax(Yp2,1))[ass]
print(aa.shape)
f,ax=plt.subplots(3)

ax[0].plot(err[ass])
ax[0].set_title("MNIST test set sorted by estimated error")
ax[0].set_ylabel("Estimated error")

ax[1].plot(np.cumsum(1.0-aa))
ax[1].set_ylabel("# errors")

ax[2].plot(np.cumsum(aa)/(np.arange(len(aa))+1))
ax[2].set_ylabel("Accuracy")


plt.show()
print(np.mean( np.argmax(y_test,1)==np.argmax(Yp2,1)))


