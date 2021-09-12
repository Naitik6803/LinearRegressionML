import numpy as np

def sigmoid_derivatives(x):
    return x*(1-x)


def sigmoid(x):
    return 1/(1+np.exp(-x))

training_inputs= np.array([[0,0,1],[1,0,1],[1,0,1],[0,1,1]])

training_outputs=np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights = 2*np.random.random((3,1))-1

print("Random starting synaptic_weights :")
print(synaptic_weights)

for i in range (10000):
    input_layer=training_inputs
    output=sigmoid(np.dot(input_layer,synaptic_weights))
    error= training_outputs-output
    adjustment=error*sigmoid_derivatives(output)
    synaptic_weights+=np.dot(input_layer.T,adjustment)

print("synaptic weights after training :")
print(synaptic_weights)

print("Output after training :")
print(output)


