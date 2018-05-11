import numpy as np

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
        self.function = hyper_parameters["function"]
        self.dropout = hyper_parameters["dropout"]
        self.drop = []

    def reset(self):
        self.hidden_layers = self.hidden_layers_initialization()
        self.forward_results = []
        self.backward_results = []
        self.loss = []

    def forward(self, x):
        lineal_dot = np.dot(x, self.hidden_layers[0])
        self.forward_results.append(self.function.activation(lineal_dot))

        n = self.hidden_layers.shape[0] - 1
        for i in range(1,n):
            #lineal_dot = np.dot(self.function.dropout(self.forward_results[-1],self.dropout), self.hidden_layers[i])
            lineal_dot = np.dot(self.forward_results[-1], self.hidden_layers[i])
            #print("shape de dop",i,lineal_dot.shape)
            self.forward_results.append(self.function.activation(lineal_dot))
        i += 1
        # lineal_dot = np.dot(self.function.dropout(self.forward_results[-1],self.dropout), self.hidden_layers[i])
        lineal_dot = np.dot(self.forward_results[-1], self.hidden_layers[i])
        #print("shape de softmax",lineal_dot.shape)
        #input("waiting...")
        return self.function.softmax(lineal_dot)

    def backward(self, o, y):
        # Calcula el loss con one hot encode de y
        #y = self.function.one_hot_encode(o.shape, y)
        self.loss.append(self.function.cross_entropy(o,y))

        # Calcula el delta
        """o_error = o - y
        o_delta =  self.loss[-1] * (o - y)"""
        #print(self.loss[-1])
        #input("sirvio....")
        y = self.function.one_hot_encode(o.shape, y)
        o_delta = self.function.cross_entropy_prime(o,y)
        #o_delta *= self.loss[-1]
        #print(o_delta.shape)
        #input()
        self.backward_results.append(o_delta)
        
        n = len(self.hidden_layers)
        for i in reversed(range(1,n)):
            #print(i)
            error = self.backward_results[-1].dot(self.hidden_layers[i].T)
            delta = error * self.function.activation_prime(self.forward_results[i-1])

            # Intentando modificar esta picha
            #error = self.backward_results[-1] * self.function.activation_prime(self.forward_results[i])
            #delta = np.dot(error, self.hidden_layers[i-1])
            self.backward_results.append(delta)

    def update(self, x):

        # Revierte los resultados del backward(derivadas o primes)
        self.backward_results = self.backward_results[::-1]

        # Actualiza el primero
        self.hidden_layers[0] += x.T.dot(self.backward_results[0]) * self.learning_rate #np.dot(np.transpose(x), self.backward_results[0]) * self.learning_rate
        n = self.hidden_layers.shape[0]

        # Actualiza todos los pesos
        for i in range(1,n):
            self.hidden_layers[i] += self.forward_results[i-1].T.dot(self.backward_results[i]) * self.learning_rate #np.dot(np.transpose(self.forward_results[i-1]), self.backward_results[i]) * self.learning_rate

        self.forward_results = []
        self.backward_results = []
    
    def train(self, x, y):
        o = self.forward(x)
        self.backward(o,y)
        self.update(x)



    def xavier_initialization(self, rows, columns):
        # Xavier initialization for a layer
         return np.random.randn(rows, columns).astype(np.float64) * np.sqrt(2.0/self.batch_size)
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
        return np.array(hidden_layers)


