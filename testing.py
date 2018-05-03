from NN import NN
#from LogLossSoftm
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import numpy as np
from LogLossSoftmax import LogLossSoftmax

# input_size = 5, output_size = 9, hidden_layers_size = [2,3,4]
network = NN()

mnist = fetch_mldata('MNIST original')    
x = mnist.data[0:10]
y = mnist.target[0:10]

hyper_parameters = {}
function = LogLossSoftmax()
hyper_parameters["input_size"] = x[0].shape[0]
hyper_parameters["output_size"] = 10
hyper_parameters["hidden_layers_size"] = [2,3]
hyper_parameters["learning_rate"] = 0.085
hyper_parameters["function"] = function
# print(hyper_parameters)

network.config(hyper_parameters)
network.forward(x)
print(function.loss(network.forward_results[-1]))

