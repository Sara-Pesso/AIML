from math import *
def neuron(w,x,b):
    neuron_weight = b
    for i in range(len(w)):
        neuron_weight += w[i]*x[i]
    return neuron_weight

def neuron(w,x,b):
    return sum([w[i]*x[i] for i in range(len(w))]) + b

#Activation Functions
# 1) Perceptrons: 
def perceptron(w,x,b):
    total_weight = sum([w[i]*x[i] for i in range(len(w))])
    if total_weight > b:
        return 1
    else:
        return 0

# x = [1,1,0]
# w = [3,2,6]
# b = 4

# print(perceptron(w,x,b))
# 2) Sigmoid:
def sigmoid(z):
    #where z = wx + b (the output of the neuron function)
    return 1/(1+e**-z)

def tanch(z):
    #where z = wx + b (the output of the neuron function)
    return (e**z - e**-z)/(e**z + e**-z)

def ReLU(z):
    #Rectified Linear Units
    return max(0,z)




