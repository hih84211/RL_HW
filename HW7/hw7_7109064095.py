#
# Quadratic Regression PyTorch "skeleton" file for homework
#
# Your task: Complete the missing code (see comments below in the code):
#            1. Quadratic regression model (w2*x**2 + w1*x + b, 3 trainable parameters)
#            2. Cost function (should use MSE)
#            3. Training loop
#            4. Plot results
#

# Description of numbered code sections below:
# 1. Generate random dataset for training
# 2. Create PyTorch quadratic regression model
# 3. Train the model
# 4. Plot fitted curve vs. data
#


import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(123)  # let's make things repeatable!

############################################
# 1. Generate the dataset
#
# create a random toy dataset for regression

np.random.seed(0)


def make_random_data():
    x = np.random.uniform(low=-2, high=4, size=200)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0,
                             scale=(0.5 + t * t / 3),
                             size=None)
        y.append(r)
    return x, 2.3 * x ** 2 + 1.726 * x - 0.84 + np.array(y)


x, y = make_random_data()

plt.plot(x, y, 'o')
plt.show()

#
#
############################################


############################################
# 2. Create the quadratic regression model
#
weight = (0.01 * torch.randn(size=(2, ), dtype=torch.float32)).clone().detach().requires_grad_(True)
bias = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# quadratic regression model
#


def quad_reg_model(feature_input):
    feature_input = np.reshape(feature_input, (-1, 1))
    feature_input = torch.tensor(feature_input, dtype=torch.float32)
    output = weight[1] * torch.pow(feature_input, 2) + weight[0] * feature_input + bias
    return output

# the MSE cost function
#


def cost(model_out, target_input):
    target_input = np.reshape(target_input, (-1, 1))
    target_input = torch.tensor(target_input, dtype=torch.float32)
    l = len(model_out)
    cost = torch.sum(torch.pow((model_out - target_input), 2)) * (1.0 / len(model_out))
    return cost


#
#
############################################


############################################
# 3. Train the model
#
optim = torch.optim.SGD([weight, bias], lr=0.01)

n_epochs = 401
training_costs = []
for e in range(n_epochs):
    optim.zero_grad()
    out = quad_reg_model(x)
    cost_tmp = cost(out, y)
    cost_tmp.backward()
    optim.step()
    training_costs.append(float(cost_tmp))
    if not e % 50:
        print('Epoch %4d: %.4f' % (e, float(cost_tmp)))

# plot cost vs. epochs
plt.plot(training_costs)
plt.show()


#
#
############################################


############################################
# 4. Plot fitted curve vs. data
#
x_fit = np.linspace(-2.5, 4.5, 20)
w = weight.detach().numpy()
b = bias.detach().numpy()

print('\nw-fit: ', w, ' b-fit: ', b)

y_fit = w[1] * x_fit**2 + w[0] * x_fit + b
plt.plot(x_fit, y_fit)
plt.plot(x, y, 'o')
plt.show()

#
#
############################################
