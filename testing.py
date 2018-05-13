from NN import NN
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import numpy as np
from Functions import Functions
import tickle
import sys
import random

mnist = fetch_mldata('MNIST original')    

x = mnist.data.astype(np.float64)
y = mnist.target.astype(np.int_)
x = x / 255
x = list(x)
y = list(y)


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
	l = []
	print("Entrenamiento")
	for e in range(epoch):
		x = train_data
		y = train_labels
		for i in range(n-1):
			i1 = batch_size * random.randint(0,n-i-1)
			i2 = i1 + batch_size
			train_x = x[i1:i2]
			train_y = y[i1:i2]
			if len(train_x) == (0):
				print("iteracion",i)
				print("len de train x",len(train_x))
				print("len de train y",len(train_y))
				print("len de x",len(x))
				print("len de y",len(y))
				print("i1",i1)
				print("i2",i2)
				input()
			network.train(train_x, train_y)
			x = x[:i1] + x[i2:]
			y = y[:i1] + y[i2:]
		print("Loss de epoch",e,"->",np.sum(np.array(network.loss))/n)
		l = l + network.loss
		network.reset()
	print("Loss total ->",np.sum(l)/train)
	
	tickle.save(network.get_config(), nombre)
else:
	print("\nClasificacion")
	n = int(test / test_size)
	for i in range(n-1):
		i1 = test_size * random.randint(0,n-i-1)
		i2 = i1 + test_size
		test_x = test_data[i1:i2]
		test_y = test_labels[i1:i2]
		network.classify(test_x, test_y)

		test_data = test_data[:i1] + test_data[i2:]
		test_labels = test_labels[:i1] + test_labels[i2:]
	print("Exactitud de la red con train","->",np.sum(np.array(network.exactitud))/n)
	network.reset()