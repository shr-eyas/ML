import numpy as np
import matplotlib.pyplot as plt

# input_array = np.array([1,2,3])
# exp_array = np.exp(input_array)

# input_val = 1  
# exp_val = np.exp(input_val)

def sigmoid(z):

    g = 1/(1+np.exp(-z)) 
    return g

z_tmp = np.arange(-10,11)

y = sigmoid(z_tmp)

np.set_printoptions(precision=3) 
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])
