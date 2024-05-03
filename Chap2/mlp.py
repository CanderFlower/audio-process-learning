from random import random
import numpy as np
from Artificial_Neuron import sigmoid

def rms(array):
    return (np.sum(array**2)/len(array))**0.5

class MLP:
    def __init__(self, num_input, num_hidden, num_output):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        layers = [num_input] + num_hidden + [num_output]
        
        self.weights = []
        for i in range(len(layers)-1):
            curr_weight = np.random.rand(layers[i], layers[i+1])
            self.weights.append(curr_weight)
            
        self.activations = []
        for i in range(len(layers)):
            curr_activation = np.zeros(layers[i])
            self.activations.append(curr_activation)    
        
        self.derivatives = []
        for i in range(len(layers)-1):
            curr_derivative = np.zeros((layers[i],layers[i+1]))
            self.derivatives.append(curr_derivative)
    
    def forward_prop(self, input):
        
        activation = input
        self.activations[0] = activation
        
        for i,w in enumerate(self.weights):
            activation = np.dot(activation, w)
            activation = sigmoid(activation)
            self.activations[i+1] = activation
        
        return activation
    
    def back_prop(self,error):
        for i in reversed(range(len(self.weights))):
            delta = error * self.activations[i+1] * (1-self.activations[i+1])
            delta = np.reshape(delta,(1,-1))
            
            curr_activation = self.activations[i]
            curr_activation = np.reshape(curr_activation,(-1,1))
            
            derivative = np.dot(curr_activation,delta)
            self.derivatives[i] = derivative
            
            error = np.dot(delta,self.weights[i].T)
            
        return error
            
    def gradient_descent(self,learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] += self.derivatives[i] * learning_rate
            
    def train(self,inputs,targets,echoes,learning_rate):
        if len(inputs)!=len(targets):
            print("length not matched!")
            return
        for i in range(echoes):
            sum_error = 0.0
            for input,target in zip(inputs,targets):
                result = self.forward_prop(input)
                error = target - result
                self.back_prop(error)
                self.gradient_descent(learning_rate)
                sum_error += rms(error)
            avg_error = sum_error / len(inputs)
            print("Echo {}: Error={}".format(i,avg_error))
    
if __name__ == "__main__":
    mlp = MLP(2,[5],1)
    inputs = np.array([[random()/2 for _ in range(2)] for _ in range(20000)])
    targets = np.array([[input[0]+input[1]] for input in inputs])
    mlp.train(inputs,targets,1500,0.1)
    
    test_inputs = np.array([[round(random()/2,1) for _ in range(2)] for _ in range(20)])
    test_targets = np.array([[test_input[0]+test_input[1]] for test_input in test_inputs])
    sum_error = 0.0
    for i,(test_input,test_target) in enumerate(zip(test_inputs,test_targets)):
        prediction = mlp.forward_prop(test_input)
        error = abs(test_target-prediction)
        sum_error += error
        print("Test#{}: {}+{}={}, Error={}".format(i,test_input[0],test_input[1],prediction,error))
    avg_error = sum_error / len(test_inputs)
    print("Average_Error={}".format(avg_error))
        