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

    def config(self, hyper_parameters):
        self.input_size = hyper_parameters["input_size"]
        self.output_size = hyper_parameters["output_size"]
        self.hidden_layers_size = hyper_parameters["hidden_layers_size"]
        self.learning_rate = hyper_parameters["learning_rate"]
        self.batch_size = hyper_parameters["batch_size"]
        self.function = hyper_parameters["function"]

        self.hidden_layers = self.hidden_layers_initialization()
        self.forward_results = []
        self.backward_results = []


    def forward(self, x):
        index = 0
        x = x.astype(np.float64)
        self.forward_results.append(np.dot(x,self.hidden_layers[index]))
        self.forward_results.append(self.function.relu(self.forward_results[index]))

        n = self.hidden_layers.shape[0] - 1
        for i in range(1,n):
            self.forward_results.append(np.dot(self.forward_results[-1],self.hidden_layers[i]))
            self.forward_results.append(self.function.relu(self.forward_results[-1]))
        i += 1
        self.forward_results.append(np.dot(self.forward_results[-1],self.hidden_layers[i]))
        self.forward_results.append(self.function.softmax(self.forward_results[-1]))

    def backward(self, x, y):
        loss = self.function.cross_entropy(self.forward_results[-1], y)
        #print(loss)

    def train(self, x, y):
        self.forward(x)
        self.backward(x, y)
        self.set_values()

        self.forward_results = []
        self.backward_results = []

    def set_values(self):
        pass

    def xavier_initialization(self, rows, columns):
        # Xavier initialization for a layer
        # return np.random.randn(rows, columns).astype(np.float64) * np.sqrt(2.0/rows)
        return np.random.randn(rows, columns).astype(np.float64) * np.sqrt(2.0/self.batch_size)

    def hidden_layers_initialization(self):
        # Layers
        hidden_layers = []

        # Layer's size
        layers = np.append(self.hidden_layers_size,[self.output_size])

        # Current layer size
        current_layer = layers[0]

        # Add the first set of w's
        hidden_layers.append(self.xavier_initialization(self.input_size, current_layer))

        # Number of hidden layer's
        n = layers.shape[0]

        # Create the w's for each layer
        for i in range(1,n-1):
            hidden_layers.append(self.xavier_initialization(current_layer, layers[i]))
            current_layer = layers[i]

        # Add the last set of w's
        hidden_layers.append(self.xavier_initialization(current_layer, self.output_size))

        # Return the weights
        return np.array(hidden_layers)


