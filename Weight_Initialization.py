## Chapter 9: Weight Initialization// Deep Learning Illustrated
## Sara Pesso 2023 09 26
#########################################################################################################################################################################################
#########################################################################################################################################################################################
# Exploring th inplications of weight and bias initializations and how to combat weak learning and neuron saturation.
# Loosely based on the MNIST DNN model from previous chapter.
#########################################################################################################################################################################################
#########################################################################################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, Activation
from keras.initializers import Zeros, RandomNormal, glorot_normal, glorot_uniform

n_input = 784
n_dense = 256

b_init = Zeros()
# sample initial weights from a standrad normal distribution about 0
w_init = RandomNormal(stddev=1.0)

model = Sequential()
model.add(Dense(n_dense,
                input_dim=n_input,
                kernel_initializer=w_init,
                bias_initializer=b_init))
model.add(Activation('sigmoid'))

#make some random "pixel values"
x = np.random.random((1,n_input))

# Forward Propogation
a = model.predict(x)

# histogeam of our activations (a)
plt.hist(np.transpose(a))
# plt.show()

# Xavier Glorot Distributions
# Glorot Normal Distributions are truncated normal distributions with mean = 0 an  stddev = sqrt(2/(n_in + n_out))
# w_init = glorot_normal()
# Glorot Uniform Distributions are [-l,l], where l = sqrt(6/(n_in+n_out))
w_init = glorot_uniform()

model = Sequential()
model.add(Dense(n_dense,
                input_dim=n_input,
                kernel_initializer=w_init,
                bias_initializer=b_init))
model.add(Activation('sigmoid'))

#make some random "pixel values"
x = np.random.random((1,n_input))

# Forward Propogation
a = model.predict(x)

# histogeam of our activations (a)
plt.hist(np.transpose(a))
# plt.show()

#########################################################################################################################################################################################
#########################################################################################################################################################################################

## Unstable Gradients
# Batch normalization: essentially normalizing the outputs of each layer before they're used in the next, which helps to mitigate the possibility
# of exploding and vanishing gradients

# batch =  activations from previous layer
# (batch - mean(batch)/stddev(batch))



