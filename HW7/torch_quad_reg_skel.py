#
# Quadratic Regression PyTorch "skeleton" file for homework
#
# Your task: Complete the missing code (see comments below in the code):
#            1. Quadratic regression model (w2*x**2 + w1*x + b, 3 trainable parameters)
#            2. Cost function (should use MSE)
#            3. Training loop
#            4. Plot results
#

#Description of numbered code sections below:
# 1. Generate random dataset for training
# 2. Create PyTorch quadratic regression model
# 3. Train the model
# 4. Plot fitted curve vs. data
#


import torch
import numpy as np

torch.manual_seed(123) #let's make things repeatable!


############################################
# 1. Generate the dataset
#
## create a random toy dataset for regression 
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

def make_random_data():
    x = np.random.uniform(low=-2, high=4, size=200)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0,
                            scale=(0.5 + t*t/3),
                            size=None)
        y.append(r)
    return  x, 2.3*x**2+1.726*x -0.84 + np.array(y)

x, y = make_random_data()

plt.plot(x, y, 'o')
plt.show()

#
#
############################################


############################################
# 2. Create the quadratic regression model
#

#quadratic regression model
#

#Fill in the missing code here!


#the MSE cost function
#

#Fill in the missing code here!

#
#
############################################


############################################
# 3. Train the model
#

#Fill in the missing code here!
 
#
#
############################################


############################################
# 4. Plot fitted curve vs. data
#

#plot fitted curve vs. data
#

#Fill in the missing code here!

#
#
############################################

