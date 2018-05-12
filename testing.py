from NN import NN
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import numpy as np
from Functions import Functions
import tickle
import sys


mnist = fetch_mldata('MNIST original')    

x = mnist.data.astype(np.float64)
y = mnist.target.astype(np.int_)

x = x / 255
batch_size = 32
test_size = 10
train = 60000
test = 10000

train_data = x[:train]
train_labels = y[:train]
test_data = x[train:]
test_labels = y[train:]

nombre = sys.argv[1]
hyper_parameters = {}
function = Functions()
#hyper_parameters = tickle(nombre)
try:
	print("*** Cargo la config ***")
	hyper_parameters = tickle.load(nombre)
	hyper_parameters["function"] = function
except:
	print("*** Creo la config ***")
	hyper_parameters["input_size"] = x[0].shape[0]
	hyper_parameters["output_size"] = 10
	hyper_parameters["hidden_layers_size"] = eval(sys.argv[4])#[32,16]
	hyper_parameters["learning_rate"] = 0.0085
	hyper_parameters["function"] = function
	hyper_parameters["batch_size"] = batch_size
	hyper_parameters["dropout"] = 0.5

entrenar = sys.argv[2]
network = NN()
network.config(hyper_parameters)

if entrenar == "entrenar":
	n = int(train / batch_size)
	epoch = int(sys.argv[3])

	print("Entrenamiento")
	for e in range(epoch):
		x = train_data
		y = train_labels
		for i in range(n):
			train_x = x[:batch_size]
			train_y = y[:batch_size]
			network.train(train_x, train_y)
			#print("Loss",i,network.loss[-1])
			x = x[batch_size:]
			y = y[batch_size:]
		#print("Loss del epoch",e,"->",np.sum(np.array(network.loss))/n)
		network.clean()
	tickle.save(network.get_config(), nombre)
else:
	print("\nClasificacion")
	n = int(test / test_size)
	for i in range(n):
		test_x = test_data[:test_size]
		test_y = test_labels[:test_size]
		network.classify(test_x, test_y)

		test_data = test_data[test_size:]
		test_labels = test_labels[test_size:]
	print("Exactitud de la red con train","->",np.sum(np.array(network.exactitud))/n)
	network.reset()