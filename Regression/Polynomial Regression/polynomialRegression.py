import matplotlib.pyplot as plt
import numpy as np

x_train = np.array([1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22])
y_train = np.array([100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100])

cost_values = []

def compute_cost(x,y,w1,w2,b):

    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w1*x[i]**2 + w2*x[i] + b
        cost = (f_wb - y[i])**2
        cost_sum = cost_sum + cost
    total_cost = (1/(2*m))*cost_sum

    return total_cost

def compute_gradient(x,y,w1,w2,b):

    m = x.shape[0]
    dj_dw1 = 0
    dj_dw2 = 0
    dj_db =  0

    for i in range(m):
        f_wb = w1*x[i]**2 + w2*x[i] + b
        dj_dw1_i = (f_wb - y[i]) * x[i]**2
        dj_dw2_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_db += dj_db_i
        dj_dw1 += dj_dw1_i
        dj_dw2 += dj_dw2_i
    dj_dw1 = dj_dw1/m
    dj_dw2 = dj_dw2/m
    dj_db = dj_db/m

    return dj_dw1,dj_dw2,dj_db    

def gradient_descent(x,y,w1_in,w2_in,b_in,alpha,num_iters,cost_function,gradient_function):
    b = b_in
    w1 = w1_in
    w2 = w2_in

    for i in range(num_iters):
        dj_dw1,dj_dw2,dj_db = compute_gradient(x,y,w1,w2,b)
        b = b - alpha*dj_db
        w1 = w1 - alpha*dj_dw1
        w2 = w2 - alpha*dj_dw2
        cost_values.append(compute_cost(x, y, w1, w2, b))
    return w1, w2, b

w1_init = 0
w2_init = 0
b_init = 0

iterations = 1000000
tmp_alpha = 0.00001

w1_final, w2_final, b_final = gradient_descent(x_train ,y_train, w1_init, w2_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)


cost_val_init = compute_cost(x_train,y_train,w1_init,w2_init,b_init)
cost_val_final = compute_cost(x_train,y_train,w1_final,w2_final,b_final)
print(cost_val_init)
print(cost_val_final)

def predict(x, w1, w2, b):
    return w1 * x**2 + w2 * x + b

x_range = np.linspace(min(x_train), max(x_train), 100)
y_predicted = predict(x_range, w1_final, w2_final, b_final)

# plotting the cost values
plt.plot(range(iterations), cost_values)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs. Iteration')
plt.show()

plt.scatter(x_train, y_train)
plt.plot(x_range, y_predicted, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.show()
