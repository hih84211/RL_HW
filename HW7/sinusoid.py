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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

torch.manual_seed(123)  # let's make things repeatable!

############################################
# 1. Generate the dataset
#
## create a random toy dataset for regression

np.random.seed(0)


def make_sinusoid_data(num=100):
    t = np.linspace(-10 * np.pi, 10 * np.pi, num)
    out_array = 1/2 * np.sin(t)
    plt.plot(t, out_array, color='red', marker="o")
    plt.title("numpy.sin()")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    return t, out_array


# Simple NN
layer1 = (1/4 * torch.randn(size=(1, 4), dtype=torch.float32)).clone().detach().requires_grad_(True)
bias1 = torch.tensor([0.0] * 4, dtype=torch.float32, requires_grad=True)
layer2 = (1/4 * torch.randn(size=(4, 4), dtype=torch.float32)).clone().detach().requires_grad_(True)
bias2 = torch.tensor([0.0] * 4, dtype=torch.float32, requires_grad=True)
layer3 = (1/4 * torch.randn(size=(4, 4), dtype=torch.float32)).clone().detach().requires_grad_(True)
bias3 = torch.tensor([0.0] * 4, dtype=torch.float32, requires_grad=True)
layer4 = (1/4 * torch.randn(size=(4, 1), dtype=torch.float32)).clone().detach().requires_grad_(True)
bias4 = torch.tensor([0.0] * 1, dtype=torch.float32, requires_grad=True)



def dnn(feature_input):
    feature_input = np.reshape(feature_input, (-1, 1))
    feature_input = torch.tensor(feature_input, dtype=torch.float32)

    hid_out = torch.tanh(torch.matmul(feature_input, layer1) + bias1)
    hid_out = torch.sinc(torch.matmul(hid_out, layer2) + bias2)
    # hid_out = torch.sin(torch.matmul(hid_out, layer3) + bias3)
    output = torch.tanh(torch.matmul(hid_out, layer4) + bias4)

    return output


loss_func = torch.nn.MSELoss()


def predict(feature_input):
    predict = dnn(feature_input)
    return predict.detach().numpy()


def cost(model_out, target_input):
    target_input = np.reshape(target_input, (-1, 1))
    target_input = torch.tensor(target_input, dtype=torch.float32)
    cost = loss_func(model_out, target_input)

    return cost


optim = torch.optim.RMSprop([layer1, bias1, layer2, bias2, layer4, bias4], lr=0.01)

data = make_sinusoid_data(800)
data_train, data_test, target_train, target_test = train_test_split(data[0], data[1], test_size=0.25,
                                                                    random_state=11)

n_epochs = 20001
training_costs = []

for e in range(n_epochs + 1):
    optim.zero_grad()
    cost_tmp = cost(dnn(data_train), target_train)
    cost_tmp.backward()
    optim.step()
    training_costs.append(float(cost_tmp))
    if not e % 50:
        print('Epoch %4d: %.6f' % (e, float(cost_tmp)))

plt.plot(training_costs)
plt.show()

pred = predict(data_train)

# impedance matching...
target_train.shape = (-1,)
pred.shape = (-1,)


x_fit = np.linspace(-35, 35, 500)
y_fit = predict(x_fit)

plt.plot(x_fit, y_fit, 'o')
plt.show()
