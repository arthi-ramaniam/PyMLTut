# modified version of code from http://www.marekrei.com/blog/theano-tutorial/

import numpy
import theano
import theano.tensor as T

# variable of the model
A = theano.tensor.dmatrix('A')
b = theano.tensor.dvector('b')
y = theano.tensor.dscalar('y')

# parameter of the model
x = theano.shared(numpy.asarray([2.1,3.9]), 'x')


# the model
hat_y  = 0.5*T.dot(T.dot(A,x),x) - T.dot(b,x)

# error
err    = theano.tensor.sqr(hat_y - y)

# gradient of error w.r.to parameter
grad   = theano.tensor.grad(err, [x])

# update
alpha  = 0.5

# gradient update
hat_x  = x - (alpha * grad[0])

updates= [(x, hat_x)]

f = theano.function([A, b, y], hat_y, updates = updates)

for i in xrange(100):
    output = f([[1.0,0.0],[0.0,1.0]],[2.0,3.0],-6.5)
    print "f = ",output, "x=",  x.eval()

