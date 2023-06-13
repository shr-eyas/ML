'''
applying neural networks to find out the optimized time and temperature for roasting coffee beans
3 neurons in first layer
1 neuron in second layer
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

'''
x = np.array([200.0, 17.0]) creates 1D vector while 
x = np.array([[200.0, 17.0]]) creates 2D (row) matrix
'''

X = np.array([[185.32, 12.69], [259.92, 11.87], [231.01, 14.41], [175.37, 11.72], [187.12, 14.13], [225.91, 12.1 ], [208.41, 14.18], [207.08, 14.03], [280.6, 14.23], [202.87, 12.25]])
Y = np.array([[1.],[0.],[0.],[0.],[1.],[1.],[0.],[0.],[0.],[1.]])

norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)

def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g

def my_dense(a_in,W,b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:,j]                        #assigns 'j'th column of matrix W to w
        z = np.dot(w, a_in) + b[j]
        a_out[j] = sigmoid(z)   
    return(a_out)

def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x,  W1, b1)
    a2 = my_dense(a1, W2, b2)
    return(a2)

# already trained data

                #    w1      w2     w3
W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], 
                    [-0.1,  -7.32, 10.81]] )

b1_tmp = np.array( [-9.82, -9.28,  0.96] )

                #    w1
W2_tmp = np.array( [[-31.18], 
                    [-27.59], 
                    [-32.56]] )

b2_tmp = np.array( [15.41] )


def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = my_sequential(X[i], W1, b1, W2, b2)
    return(p)

X_tst = np.array([
    [200,13.9],         # negative example
    [200,17]])          # negative example
X_tstn = norm_l(X_tst)  # normalize
predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
        
print(f"decisions = \n{yhat}")
