# -*- coding: utf-8 -*-
"""
@author: Oguzhan Calikkasap
Student ID: 150714822

"""

import numpy as np

    
traindata = np.loadtxt('deneme.txt')
# Split features and labels.
feature_matrix, class_labels = traindata[:, :-1], traindata[:, -1]


# Activation functions.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def ReLu(x):
    return np.maximum(0, x)

#Gradient of sigmoid
def sigmoidPrime(x):
    return np.exp(-x)/((1+np.exp(-x))**2)

# Loss function: Cross-entropy
def CrossEntropy(predictions, targets, epsilon=1e-12):
    
    # Make sure the predictions between 0 and 1
    #predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions))/N
    return ce


# Softmax function.
def softmax(out):
    exp = np.exp(out)
    return (np.exp(out)) / (np.sum(exp))


# Create input nodes as many as the number of features.
def inputLayer(feature_matrix): 
    return np.zeros(feature_matrix.shape[1])

# Store entered amount of hidden layers with entered number of nodes.
def hiddenLayer(num_of_layers=1, nodes_per_layer=[5]):
    
    # List to hold hidden layers.
    hidden_layers = []
    
    if len(nodes_per_layer) != num_of_layers:
        print('Error, you should specify the number of nodes for each hidden layer!')
    else:
        for i in range (num_of_layers):
            hidden_layers.append(np.zeros(nodes_per_layer[i]))
                
    return hidden_layers


# Determine the size and number of weight matrices.
def Weights(input_layer, hidden_layers, out_layer_size):
    
    Weights = []
    
    for i in range (len(hidden_layers)):
        
        if i == 0: # First iteration, check with input layer.
            Weights.append(np.random.uniform(-1, 1, (input_layer.shape[0], hidden_layers[i].size)))
        # Now all we need to check is the hidden layers.
        else:
            Weights.append(np.random.uniform(-1, 1, (hidden_layers[i-1].size, hidden_layers[i].size)))
    
    Weights.append(np.random.uniform(-1, 1, (hidden_layers[-1].size, out_layer_size)))
        
    return Weights


def feedforward(input_layer, hidden_layers, weights):
    

    for i in range (len(hidden_layers)):
        
        # First iteration is between the input and first hidden layer.
        if i == 0:
            hidden_layers[i] = sigmoid(np.dot(weights[i].T, input_layer))
        # Next ones are between only hidden layers.
        else:
            hidden_layers[i] = sigmoid(np.dot(weights[i].T, hidden_layers[i-1]))
    # Last iteration is the final feedforward output of the network.
    out = np.dot(weights[-1].T, hidden_layers[-1])

    return out





    

    
    
    
    
    
    
    
    
    
    
    
    
    