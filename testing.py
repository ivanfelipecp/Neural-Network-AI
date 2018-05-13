from NN import NN
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import numpy as np
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

try:
	hyper_parameters = tickle.load(nombre)
	print("*** Cargo la config ***")
except:
	
	hyper_parameters["input_size"] = x[0].shape[0]
	hyper_parameters["output_size"] = 10
	hyper_parameters["hidden_layers_size"] = eval(sys.argv[1])#[32,16]
	hyper_parameters["learning_rate"] = float(sys.argv[2])
	hyper_parameters["batch_size"] = batch_size
	hyper_parameters["dropout"] = float(sys.argv[3])
	print("*** Creo la config ***")

entrenar = sys.argv[6]
network = NN()
network.config(hyper_parameters)


if entrenar == "entrenar":
	n = int(train / batch_size)
	epoch = int(sys.argv[4])
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
			network.train(train_x, train_y)
			x = x[:i1] + x[i2:]
			y = y[:i1] + y[i2:]
		#c += np.sum(np.array(network.loss)) / n
		print("Loss de epoch",e,"->",np.sum(np.array(network.loss)) / n)
		l = l + network.loss
		network.reset()
	print("Loss total ->",np.sum(l)/train)
	print("Loss de c ->",c/epoch)
	
	tickle.save(network.get_config(), nombre)
else:
	print("Clasificacion")
	n = int(test / test_size)
	x = test_data
	y = test_labels
	for i in range(n-1):
		i1 = test_size * random.randint(0,n-i-1)
		i2 = i1 + test_size
		x = test_data[i1:i2]
		y = test_labels[i1:i2]
		network.classify(x, y)

		test_data = test_data[:i1] + test_data[i2:]
		test_labels = test_labels[:i1] + test_labels[i2:]
	print("Exactitud de la red con train","->",np.sum(np.array(network.exactitud))/n)
	network.reset()