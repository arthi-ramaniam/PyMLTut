# modified code from https://keras.io/getting-started/functional-api-guide/

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD 
import h5py

# loading data
f  = h5py.File("auto-mpg.hdf5", "r")
D  = f['dataset'][...]
X  = D[:,1:8]
Y  = D[:,0]

'''
mu,var = X.mean(),X.var()
X = (X - mu)/var
mu,var = Y.mean(),Y.var()                                                                                                                     
Y = (Y - mu)/var
#'''

# this returns a tensor
inputs = Input(shape=(7,))

# a layer instance is callable on a tensor, and returns a tensor
a1 = Dense(10, activation='sigmoid')(inputs)
a2 = Dense(1, activation='tanh')(a1)

# optimizer
sgd = SGD(lr=0.001)

# model
model = Model(input=inputs, output=a2)
model.compile(optimizer=sgd, loss='mse')

# training
model.fit(X, Y, batch_size=16)
