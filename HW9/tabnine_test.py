import numpy as np
import torch

x1 = torch.rand(2, 3, dtype=torch.float32)
x2 = np.random.rand(2, 3)
x3 = torch.tensor(x2, dtype=torch.float32)
x4 = x3.numpy()
x3 = x3+1
print(x2)
print(x4)
