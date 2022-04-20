#
# Simple example to demonstrate PyTorch variables and expressions 
#

import torch
import numpy as np

a = torch.tensor([2., 3., 4.], requires_grad=True)
b = torch.tensor([4., 5., 6.], requires_grad=True)


def compute(x):
    x = torch.tensor(x, dtype=torch.float32)
    output = torch.sum(a * x + b)
    return output


def compute2(x):
    x = torch.tensor(x, dtype=torch.float32)
    output = torch.sum(a * x - b * x)
    return output


result = compute([7., 8., 9.])
result.backward()  # compute gradients on model graph

# print gradients & result
print('a.grad: ', a.grad)
print('b.grad: ', b.grad)
print(result)

result = compute2([0.3, 0.1, 2.5])
result.backward()  # compute gradients on model graph

# print gradients & result
print('a.grad: ', a.grad)
print('b.grad: ', b.grad)
print(result)
#
# Can easily convert tensors to & from numpy arrays.
#
c = torch.tensor([9., 10., 11.])
d = torch.tensor([12., 13., 14.], requires_grad=True)

print(c.numpy())
print(d.detach().numpy())  # if requires_grad=True, must first detach


e = np.array([15., 16., 17.], dtype=np.float32)
print(e)
f = torch.from_numpy(e)
print(f)
