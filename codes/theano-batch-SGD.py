# modified version of code from http://www.marekrei.com/blog/theano-tutorial/

import numpy
import theano
import theano.tensor as T
import numpy.random as R
import h5py


R.seed(2442)

f  = h5py.File("auto-mpg.hdf5", "r")
D  = f['dataset'][...]
X  = D[:,1:8]
Y  = D[:,0]
print X.shape
print Y.shape

# variable of the model
x  = T.matrix('x')
y  = theano.tensor.dvector('y')

# parameter of the model
W1 = theano.shared(numpy.random.randn(7,4), 'W1')
b1 = theano.shared(numpy.random.randn(4), 'b1')
W2 = theano.shared(numpy.random.randn(4,1), 'W2')
b2 = theano.shared(numpy.random.randn(1), 'b2')

params = [W1,W2,b1,b2]


# the model

z1 = T.dot(x,W1) + b1
a1 = T.nnet.sigmoid(z1)

z2 = T.dot(a1,W2) + b2
a2 = T.nnet.sigmoid(z2)


hat_y  =  a2

# error
err    = T.mean(T.sqrt(T.sum(T.square(hat_y.T-y),axis=0)))

# gradient of error w.r.to parameter
grad   = T.grad(err, params)

# update
alpha  = 0.001

# gradient update
hat_x  = x - (alpha * grad[0])

updates= [(param,param-(alpha*deltaG)) for param,deltaG in zip(params,grad) ]

f = theano.function([x, y], [hat_y,err], updates = updates)


batch_size = 50
epochs = 1000

for i in xrange(epochs):
    error = []
    #for x,y in zip(X,Y):
    for i in range(0,X.shape[0],batch_size):
        j = max(i+batch_size,X.shape[0])
        _,erro = f(X[i:j,:],Y[i:j])
        error.append(erro)
    print "error = ",numpy.asarray(error).mean()

