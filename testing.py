from NN import NN
#from LogLossSoftm
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import numpy as np
from Functions import Functions

# input_size = 5, output_size = 9, hidden_layers_size = [2,3,4]
network = NN()

mnist = fetch_mldata('MNIST original')    
fin = 2
x = mnist.data[0:fin].astype(np.float64)
y = mnist.target[0:fin].astype(np.int_)

x = x / 255

hyper_parameters = {}
function = Functions()
hyper_parameters["input_size"] = x[0].shape[0]
hyper_parameters["output_size"] = 10
hyper_parameters["hidden_layers_size"] = [200,100]
hyper_parameters["learning_rate"] = 0.085
hyper_parameters["function"] = function
hyper_parameters["batch_size"] = fin
hyper_parameters["dropout"] = 0.5
# print(hyper_parameters)

network.config(hyper_parameters)
network.reset()
network.train(x,y)
#o = network.forward(x)
#print(np.sum(o,axis=1))
#for i in network.forward_results:
    #print(i.shape)
