'''
implementing gradient descent in a multiple regression model
'''
import math, copy
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 0.0
w_init = np.array([ 0.0, 0.0, 0.0, 0.0])

def compute_cost(x,y,w,b):

    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(x[i],w)+b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost/(2*m)
    return cost

def compute_gradient(x,y,w,b):

    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        f_wb = np.dot(x[i],w) + b
        dj_db_i = f_wb - y[i]
        for j in range(m):
            dj_dw[j] = dj_dw[j] + (f_wb - y[i])*x[i,j]
        dj_db +=  dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    b = b_in
    w = w_in

    for i in range(num_iters):
        dj_db,dj_dw = gradient_function(x, y, w, b)      
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
    return w, b
      
initial_w = np.zeros_like(w_init)
initial_b = 0.0

iterations = 1000
alpha = 5.0e-7

w_final, b_final = gradient_descent(x_train, y_train, initial_w, initial_b,compute_cost, compute_gradient, alpha, iterations)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = x_train.shape
for i in range(m):
    print(f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
