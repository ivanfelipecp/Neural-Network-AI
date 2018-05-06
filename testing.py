from NN import NN
#from LogLossSoftm
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import numpy as np
from Functions import Functions

# input_size = 5, output_size = 9, hidden_layers_size = [2,3,4]
network = NN()

mnist = fetch_mldata('MNIST original')    
batch_size = 32
x = mnist.data.astype(np.float64)
y = mnist.target.astype(np.int_)

x = x / 255

hyper_parameters = {}
function = Functions()
hyper_parameters["input_size"] = x[0].shape[0]
hyper_parameters["output_size"] = 10
hyper_parameters["hidden_layers_size"] = [10,8]
hyper_parameters["learning_rate"] = 0.0085
hyper_parameters["function"] = function
hyper_parameters["batch_size"] = batch_size
hyper_parameters["dropout"] = 0.5

network.config(hyper_parameters)
network.reset()
#network.train(x,y)

for i in range(1000):
    test_x, test_y, x, y = function.get_random_elements(batch_size, x, y)
    network.train(x,y)
    print("Loss",i,"->",network.log_loss[-1])