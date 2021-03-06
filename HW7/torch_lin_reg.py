#
# Linear Regression PyTorch example
#

#
# 1. Generate random dataset for training
# 2. Create PyTorch linear regression model
# 3. Train the model
# 4. Plot fitted curve vs. data
#


import torch
import numpy as np

torch.manual_seed(123)  # let's make things repeatable!

############################################
# 1. Generate the dataset
#
# create a random toy dataset for regression
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


def make_random_data():
    x = np.random.uniform(low=-2, high=4, size=200)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0,
                             scale=(0.5 + t * t / 3),
                             size=None)
        y.append(r)
    return x, 1.726 * x - 0.84 + np.array(y)


x, y = make_random_data()

plt.plot(x, y, 'o')
plt.show()

#
#
############################################


############################################
# 2. Create the linear regression model
#
weight = (0.25 * torch.randn(size=(1,), dtype=torch.float32)).clone().detach().requires_grad_(True)
bias = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


# the basic lin-reg model
def lin_reg_model(feature_input):
    feature_input = torch.tensor(feature_input, dtype=torch.float32)

    output = weight * feature_input + bias

    return output


# the MSE cost function
#  (note, we're implementing this manually, but we could also use torch.nn.MSELoss here)
def cost(model_out, target_input):
    target_input = torch.tensor(target_input, dtype=torch.float32)
    cost = torch.sum(torch.pow((model_out - target_input), 2)) * (1.0 / len(model_out))

    return cost


#
#
############################################


############################################
# 3. Train the model
#

# instantiate the optimizer
optim = torch.optim.SGD([weight, bias], lr=0.01)

# train the model for n_epochs
#
n_epochs = 401
training_costs = []
for e in range(n_epochs):
    optim.zero_grad()  # zero out gradient accumulation each epoch
    cost_tmp = cost(lin_reg_model(x), y)  # compute cost
    cost_tmp.backward()  # compute gradients on model graph
    optim.step()  # move optimizer forward one step
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

# plot fitted curve vs. data
x_fit = np.linspace(-2.0, 4.0, 10)
w = weight.detach().numpy()[0]
b = bias.detach().numpy()

# print the final estimated w & b model values
print()
print('w-fit: {}  b-fit: {}'.format(w, b))

# plot the results
y_fit = w * x_fit + b
plt.plot(x_fit, y_fit)
plt.plot(x, y, 'o')
plt.show()

#
#
############################################
