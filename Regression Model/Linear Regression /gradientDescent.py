import math, copy
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])               #(size in 1000 square feet)
y_train = np.array([300.0, 500.0])           #(price in 1000s of dollars)


def compute_cost(x,y,w,b):

    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        cost = ((w*x[i] + b) - y[i])**2
        cost_sum = cost_sum + cost
    total_cost = (1/2*m)*cost_sum

    return total_cost

def compute_gradient(x,y,w,b):

    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w*x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw,dj_db

def gradient_descent(x,y,w_in,b_in,alpha,num_iters,cost_function,gradient_function):
    b = b_in
    w = w_in

    for i in range(num_iters):
        dj_dw,dj_db = compute_gradient(x,y,w,b)
        b = b - alpha*dj_db
        w = w - alpha*dj_dw
    return w, b

w_init = 0
b_init = 0

iterations = 10000
tmp_alpha = 1.0e-2

w_final, b_final = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)

print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
