'''
Implementing cost function
'''
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])               #(size in 1000 square feet)
y_train = np.array([300.0, 500.0])           #(price in 1000s of dollars)

# Model parameters 
w = 200
b = 100

def compute_cost(x,y,w,b):

    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        cost = ((w*x[i] + b) - y[i])**2
        cost_sum = cost_sum + cost
    total_cost = (1/2*m)*cost_sum

    return total_cost

total = compute_cost(x_train,y_train,w,b)

print(total)
