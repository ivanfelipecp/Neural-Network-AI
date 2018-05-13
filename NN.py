import numpy as np
from Functions import Functions
class NN():
    def __init__(self):

        # Hyperparameters of the  NN
        self.input_size = None
        self.output_size = None
        self.hidden_layers_size = None 
        self.learning_rate = None

        self.hidden_layers = None 
        self.function = None
        self.forward_results = None 
        self.backward_results = None
        self.loss = None
        self.drop = None

    def config(self, hyper_parameters):
        self.input_size = hyper_parameters["input_size"]
        self.output_size = hyper_parameters["output_size"]
        self.hidden_layers_size = hyper_parameters["hidden_layers_size"]
        self.learning_rate = hyper_parameters["learning_rate"]
        self.batch_size = hyper_parameters["batch_size"]
        #self.function = hyper_parameters["function"]
        self.dropout = hyper_parameters["dropout"]
        self.function = Functions()
        self.drop = []
        self.exactitud = []
        self.loss = []
        self.forward_results = []
        self.backward_results = []

        if "hidden_layers" in hyper_parameters.keys():
            self.hidden_layers = hyper_parameters["hidden_layers"]
        else:
            self.hidden_layers = self.hidden_layers_initialization()

    def get_config(self):
        d = {}
        d["input_size"] = self.input_size
        d["output_size"] = self.output_size
        d["hidden_layers"] = self.hidden_layers
        d["hidden_layers_size"] = self.hidden_layers_size
        d["learning_rate"] = self.learning_rate
        d["batch_size"] = self.batch_size
        #d["function"] = self.function
        d["dropout"] = self.dropout
        return d

    def forward(self, x):
        lineal_dot = np.dot(x, self.hidden_layers[0])
        lineal_dot = self.function.activation(lineal_dot)
        #drop = self.function.dropout(lineal_dot, self.dropout)
        #lineal_dot *= drop
        self.forward_results.append(self.function.activation(lineal_dot))
        #self.drop.append(drop)

        n = len(self.hidden_layers) - 1
        for i in range(1,n):
            lineal_dot = np.dot(self.forward_results[-1], self.hidden_layers[i])
            lineal_dot = self.function.activation(lineal_dot)
            #drop = self.function.dropout(lineal_dot, self.dropout)
            #lineal_dot *= drop
            self.forward_results.append(lineal_dot)
            #self.drop.append(drop)
        i += 1
        lineal_dot = np.dot(self.forward_results[-1], self.hidden_layers[i])
        lineal_dot = self.function.softmax(lineal_dot)
        return lineal_dot

    def backward(self, o, y):
        # El "y" se convierte en one hot encode
        #print(type(o),type(y))
        #input()
            
        #y = self.function.one_hot_encode(o.shape, y)
        loss = self.function.cross_entropy(o,y)
        # Se calcula el loss, se agrega y se multiplica por la derivada y se agrega
        
        o_delta = loss * self.function.cross_entropy_prime(o,y)
        self.loss.append(loss)
        self.backward_results.append(o_delta)
        
        
        n = len(self.hidden_layers)
        # Back prop
        for i in reversed(range(1,n)):
            # Derivada por capa actual
            error = self.backward_results[-1].dot(self.hidden_layers[i].T)
            # Error por la derivada de la activacion
            delta = (error * self.function.activation_prime(self.forward_results[i-1])) #* self.drop[i-1]
            # Se añade a los pesos del backward
            self.backward_results.append(delta)

    def update(self, x):
        # Revierte los resultados del backward(derivadas o primes)
        self.backward_results = self.backward_results[::-1]

        # Actualiza el primero
        self.hidden_layers[0] += x.T.dot(self.backward_results[0]) * self.learning_rate #np.dot(np.transpose(x), self.backward_results[0]) * self.learning_rate

        # Actualiza todos los pesos
        n = len(self.hidden_layers)
        for i in range(1,n):
            self.hidden_layers[i] += self.forward_results[i-1].T.dot(self.backward_results[i]) * self.learning_rate #np.dot(np.transpose(self.forward_results[i-1]), self.backward_results[i]) * self.learning_rate

    def clean(self):
        self.forward_results = []
        self.backward_results = []
        self.drop = []
        
    def reset(self):
        self.loss = []
        self.exactitud = []
    
    def train(self, x, y):
        x = np.array(x)
        y = np.array(y)

        o = self.forward(x)
        self.backward(o,y)
        self.update(x)
        self.clean()
        #self.exactitud.append(self.function.exactitud(o,y))

    def classify(self, x, y):
        o = self.forward(x)
        exactitud = self.function.exactitud(o,y)
        self.exactitud.append(exactitud)
        self.clean()


    def xavier_initialization(self, rows, columns):
        # Xavier initialization for a layer
        # con esta esta funcando
        return np.random.randn(rows, columns) / np.sqrt(rows)#.astype(np.float64) * np.sqrt(2.0/self.batch_size)
        #return np.random.randn(rows, columns).astype(np.float64) * np.sqrt(2.0/rows)
        #return np.random.normal(0, 0.01,(rows, columns))

    def hidden_layers_initialization(self):
        # Layers
        hidden_layers = []

        # Layer's size [2,10]
        layers = np.append(self.hidden_layers_size,[self.output_size])

        # Current layer size
        current_layer = layers[0]

        # Add the first set of w's
        hidden_layers.append(self.xavier_initialization(self.input_size, current_layer))

        # Number of hidden layer's
        n = layers.shape[0]

        # Create the w's for each layer
        for i in range(1,n):
            hidden_layers.append(self.xavier_initialization(current_layer, layers[i]))
            current_layer = layers[i]

        # Return the weights
        return hidden_layers


