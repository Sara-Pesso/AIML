import numpy as np
import pandas as pd
import os
np.random.seed(10)
dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)
data = pd.read_csv(dir + r"\hamburger_hotdog_data.csv")

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets


data_train = data.T
Y_train = data_train[2:]
X_train = data_train[0:2]
_,m_train = X_train.shape

def initial_parameters():
    W1 = np.random.rand(2,2) - 0.5
    W2 = np.random.rand(2,2) - 0.5
    B1 = np.random.rand(2,1) - 0.5
    B2 = np.random.rand(2,1) - 0.5
    return W1, B1, W2, B2

def ReLU(Z):
    return np.maximum(Z,0)

def softmax(Z):
    Z = Z.T
    for i in range(len(Z)):
        Z[i] = np.exp(Z[i])/np.exp(Z[i]).sum()
    A = Z.T
    return A

def ln_cost(A,Y):
    return -Y*np.log(A) + (1-Y)*np.log(1-A)

def cost_deriv(A, Y):
    return np.divide(-Y,A) + np.divide((1-Y),(1-A))

def SM_deriv(A):
    return np.multiply(A, (1-A))

def ReLU_deriv(Z):
    return Z > 0

# W1, W2 ~ 2x2
# B1, B2 ~ 2x1
# X ~ 2x57 (2xm)
def forward_prop(W1, B1, W2, B2, X):
    Z1 = np.matrix(W1)*np.matrix(X) + B1 # ~ 2 x 57 (2 x m)
    A1 = ReLU(Z1) 
    Z2 = np.matrix(W2)*np.matrix(A1) + B2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):

    dZ2 = 2*(A2 - Y)

    dW2 = 1/m * A1*dZ2.T #dC/dW2
    dB2 = dZ2.mean(axis = 1)#~ 2 x 1
   
    dZ1 = np.multiply((dZ2.T*W2).T, ReLU_deriv(Z1)) #dc/dZ2 #DELTA 2

    dW1 = 1/m * X*dZ1.T
    dB1 = dZ1.mean(axis = 1)
    
    return dW2, dW1, dB1, dB2

def update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, eta):
    # print(W1)
    W1 -= eta * dW1
    B1 -= eta * dB1    
    W2-= eta * dW2  
    B2 -= eta * dB2    
    return W1, B1, W2, B2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, eta, iterations):
    W1, B1, W2, B2 = initial_parameters()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, B1, W2, B2, X)
        dW2, dW1, dB1, dB2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, B1, W2, B2 = update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, eta)

        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            acc = get_accuracy(predictions, Y[1])
            print(acc)

        
        if acc > 0.98:
            break
            
    return W1, B1, W2, B2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

