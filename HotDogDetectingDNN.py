from Neurons_And_ActivationFunctions import *
from random import seed, uniform
seed(19)

x1 = 3 #mL of ketchup
x2 = 2 #mL of mustard 


class Neuron():
    def __init__(self, w = None, x = None, b = None):
        ''' constructor ''' 
        # Generic neuron in any layer 
        self.w = w
        self.x = x
        self.b = b
    
        self.z = sum([self.w[i]*self.x[i] for i in range(len(self.w))]) + self.b

    #Activation Functions
    # 1) Perceptrons: 
    def perceptron(self):
        total_weight = sum([self.w[i]*self.x[i] for i in range(len(self.w))])
        if total_weight > self.b:
            return 1
        else:
            return 0

    # 2) Sigmoid:
    def sigmoid(self):
        #where z = wx + b (the output of the neuron function)
        return 1/(1+e**-self.z)

    def tanch(self):
        #where z = wx + b (the output of the neuron function)
        return (e**self.z - e**-self.z)/(e**self.z + e**-self.z)

    def ReLU(self):
        #Rectified Linear Units
        return max(0,self.z)


class Layer():
    def __init__(self, layer_size = None, previous_layer = None):
        self.layer_size = layer_size
        self.previous_layer = previous_layer
    
    def compute_ReLU_activations(self):
        self.a = []
        for _ in range(self.layer_size):
            self.w = [uniform(-1.5,1.5) for _ in self.previous_layer]
            self.b = uniform(-1.5,1.5)
            self.z = neuron(self.w, self.previous_layer, self.b)
            self.a.append(ReLU(self.z))
        return self.a
    
    def compute_sigmoid_activations(self):
        a = self.compute_ReLU_activations()[0]
        self.a = 1/(1+e**-a)
        return self.a

a = Layer(3, [4,4]).compute_ReLU_activations()
print(a)
b = Layer(2, a).compute_ReLU_activations()
print(b)
c = Layer(1, b).compute_sigmoid_activations()
print(c)
    
class DNN():
    def __init__(self, input_layer, hidden_layer_sizes):
        self.input_layer = input_layer
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_hidden_layers = len(hidden_layer_sizes)
        return
        
    
# =======================================================================================================================================================================================================================================================================================================================================================================================
# =======================================================================================================================================================================================================================================================================================================================================================================================
## Procedural Version: Shallow DNN
# =======================================================================================================================================================================================================================================================================================================================================================================================
# =======================================================================================================================================================================================================================================================================================================================================================================================
x = [[4,3]]
hidden_layer_sizes = [3,2]


for j in range(len(hidden_layer_sizes)):
    hidden_layer = []
    for i in range(hidden_layer_sizes[j]):
        z = neuron([uniform(-1.5,1.5) for _ in x[j]], x[j], uniform(-1.5,1.5))
        a = ReLU(z)
        hidden_layer.append(a)
    x.append(hidden_layer)

prediction = sigmoid(neuron([uniform(-1.5,1.5) for _ in x[-1]], x[-1], uniform(-1.5,1.5)))
# print(prediction)




        
