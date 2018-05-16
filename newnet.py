from NN import NN
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import numpy as np
import tickle
import sys
import random
import matplotlib.pyplot as plt

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

name = sys.argv[5]
hyper_parameters = {}



try:
	hyper_parameters = tickle.load(name)
	print("*** Cargo la config ***")
except:
	
	hyper_parameters["input_size"] = x[0].shape[0]
	hyper_parameters["output_size"] = 10
	hyper_parameters["hidden_layers_size"] = eval(sys.argv[1])#[32,16]
	hyper_parameters["learning_rate"] = float(sys.argv[2])
	hyper_parameters["batch_size"] = batch_size
	hyper_parameters["dropout"] = float(sys.argv[3])
	print("*** Creo la config ***")

network = NN()
network.config(hyper_parameters)


def plot_graphic(name, loss, eficiencia):        
        range_of = list(range(len(eficiencia)))
        fig1 = plt.figure(figsize = (8,8))
        plt.subplots_adjust(hspace=0.4)
        
        p1 = plt.subplot(2,1,1)
        l1 = plt.plot(range_of, eficiencia, 'g-')
        xl = plt.xlabel('Epoch n')
        yl = plt.ylabel('Exactitud(%)')
        grd = plt.grid(True)

        p2 = plt.subplot(2,1,2)
        ll2 = plt.plot(range_of, loss, 'c-')
        xxl = plt.xlabel('Epoch n')
        yyl = plt.ylabel('Loss')
        grd1 = plt.grid(True)

        sttl = plt.suptitle("Grafiquita de "+name)
        plt.savefig("./" + name + '.png') # Guarda la imagen de la gráfica en el Desktop.
        fig1.clf()
        #plt.show() #Esta función muestra la gráfica en ejecución, la ejecución se detiene hasta que se cierre la ventana.


def clasificar():
	global test_data, test_labels, batch_size, network, test, test_size
	n = int(test / test_size)
	x = test_data
	y = test_labels
	network.train_state = False
	for i in range(n-1):
		i1 = test_size * random.randint(0,n-i-1)
		i2 = i1 + test_size
		x = test_data[i1:i2]
		y = test_labels[i1:i2]
#		try:
		network.classify(x, y)
#		except:
		#print(i)
		#input()

		x = x[:i1] + x[i2:]
		y = y[:i1] + y[i2:]
	return np.sum(np.array(network.accuracy))/n

def entrenar():
	print("*** Entrenando ... ***")
	global train_data, train_labels, batch_size, network, name, train
	n = int(train / batch_size)
	epoch = int(sys.argv[4])
	loss_epoch = []
	
	accuracy_epoch = []
	for e in range(epoch):
		x = train_data
		y = train_labels
		for i in range(n-1):
			i1 = batch_size * random.randint(0,n-i-1)
			i2 = i1 + batch_size
			train_x = x[i1:i2]
			train_y = y[i1:i2]
			#print(i)
			network.train(train_x, train_y)
			x = x[:i1] + x[i2:]
			y = y[:i1] + y[i2:]
		e_loss = np.sum(np.array(network.loss)) / n
		loss_epoch.append(np.sum(np.array(network.loss)) / n)
		accuracy_epoch.append(clasificar())
		network.reset()
		network.train_state = True
	plot_graphic(name, loss_epoch, accuracy_epoch)
	tickle.save(network.get_config(), name)
	print("*** Red guardada como "+name+" ***")

if sys.argv[6] == "entrenar":
	entrenar()
	"""n = int(train / batch_size)
	epoch = int(sys.argv[4])
	loss_epoch = []
	loss_total = []
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
		e_loss = np.sum(np.array(network.loss)) / n
		print("Loss de epoch",e,"->",e_loss)
		loss.epoch.append(e_loss)
		loss_total += network.loss
		network.reset()
		# validación

	#print("Loss total ->",np.sum(loss_total)/train)
	#print("Loss de c ->",c/epoch)
	
	tickle.save(network.get_config(), name)
	"""
else:
	clasificar()
	"""
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
	print("Exactitud de la red con train","->",np.sum(np.array(network.accuracy))/n)
	#
	network.reset()
	"""