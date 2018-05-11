from NN import NN
#from LogLossSoftm
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import numpy as np
from Functions import Functions


mnist = fetch_mldata('MNIST original')    

x = mnist.data.astype(np.float64)
y = mnist.target.astype(np.int_)

x = x / 255
batch_size = 32
train = 60000

train_data = x[:train]
train_labels = y[:train]
test_data = x[train:]
test_labels = x[train:]

hyper_parameters = {}
function = Functions()
hyper_parameters["input_size"] = x[0].shape[0]
hyper_parameters["output_size"] = 10
hyper_parameters["hidden_layers_size"] = [32,16]
hyper_parameters["learning_rate"] = 0.0085
hyper_parameters["function"] = function
hyper_parameters["batch_size"] = batch_size
hyper_parameters["dropout"] = 0.5

network = NN()
network.config(hyper_parameters)
network.reset()

n = int(train / batch_size)
epoch = 2

#a = np.array([[1],[2],[3],[4],[5],[6]])
#print(function.dropout(a,0.5))


for e in range(epoch):
	x = train_data
	y = train_labels
	for i in range(n):
		train_x = x[:batch_size]
		train_y = y[:batch_size]
		network.train(train_x, train_y)
		print("Loss",i,network.loss[-1])
		x = x[batch_size:]
		y = y[batch_size:]
	print("Loss del epoch",i,np.sum(np.array(network.loss))/train)
	input()
print("fin")

#network.train(test_x,test_y)
#print("Loss",i,"->",network.loss)