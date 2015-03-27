import theano
from theano import tensor as T
import numpy as np

# convert to correct dtype
def floatX(X):
	return np.asarray(X,dtype=theano.config.floatX) 

# initialize model params (small gaussian distribution)
def init_weight(shape):
	return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def sgd(cost, params, lr=0.05):
	grads = T.grad(cost=cost, wrt=params)
	updates=[]
	for p, g in zip(params, grads):
		updates.append([p, p-g * lr])
	return updates


# our model in matrix form
def model(X1, X2, w_h, w_o):
	X12 = T.concatenate([X1,X2],axis=0)
	d1 = T.dot(T.transpose(X12), w_h)
	h = T.nnet.sigmoid(d1)
	d2 = T.dot(h, w_o)
	a = T.nnet.softmax(d2)
	return a

# matrix types
X1 = T.fmatrix()
X2 = T.fmatrix()
Y = T.fmatrix()

sz_input = 300 # for 100 x 3 points
w_h = init_weight((sz_input*2, 300))
w_o = init_weight((300, 2))

a = model(X1,X2,w_h,w_o)
# probability output and maxima predictions
y = T.argmax(a,axis=1)

# classification metric to optimize
cost = T.mean(T.nnet.categorical_crossentropy(a, Y))
params = [w_h, w_o]
updates = sgd(cost, params)

train = theano.function(inputs=[X1,X2,Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X1,X2], outputs=y, allow_input_downcast=True) # compile prediction function



# generate training data
trX1 = np.ones( (300, 200) )
trX2 = np.ones( (300, 200) )

trX1[:,100:200] = trX1[:,100:200] * 0.1;
trX2[:,100:200] = trX2[:,100:200] * 0.1;

# set y desired output for each training feature (for simple binary classification)
y = np.zeros( (200,2) )
for i in range(0,200):
	if(i<100):
		y[i][0]=0
		y[i][1]=1
	else:	
		y[i][0]=1
		y[i][1]=0


sz_batch = 20
cost = T.scalar()
for i in range(0,50):
	for start, end in zip(range(0,len(trX1), sz_batch) , range(sz_batch, len(trX2), sz_batch)):
		cost=train(trX1[:,start:end], trX2[:,start:end], y[start:end,:])
	print i, np.mean(np.argmax(y, axis=1) == predict(trX1,trX2))	
	

# manual test!
teX1 = np.ones( (300, 1) )
teX2 = np.ones( (300, 1) )
print predict(teX1,teX2)

teX1 = np.ones( (300, 1) ) * 0.1
teX2 = np.ones( (300, 1) ) * 0.1
print predict(teX1,teX2)

teX1 = np.ones( (300, 1) )
teX2 = np.ones( (300, 1) )
print predict(teX1,teX2)

teX1 = np.ones( (300, 1) ) * 0.1
teX2 = np.ones( (300, 1) ) * 0.1
print predict(teX1,teX2)


teX1 = np.ones( (300, 1) ) * 0.1
teX2 = np.ones( (300, 1) ) * 0.1
print predict(teX1,teX2)

teX1 = np.ones( (300, 1) )
teX2 = np.ones( (300, 1) )
print predict(teX1,teX2)


