'''
Linear regression in one variable problem
x_train is the input variable (size in 1000 square feet)
y_train is the target (price in 1000s of dollars)
Fit a linear regression model through these two points
Then predict price for other houses - say, a house with 1200 sqft
'''

import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

# m is the number of training examples
m = x_train.shape[0]

# Model parameters
w = 200
b = 100

def compute_model_output(x, w, b):

    m = x.shape[0]                  # Number of data points
    f_wb = np.zeros(m)              # Declares an array to store model output 
    for i in range(m):
        f_wb[i] = w*x[i] + b
    return f_wb

t_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, t_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()


x_i = 1.2

cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")
