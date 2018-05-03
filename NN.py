import numpy as np

class NN():
    def __init__(self, input_size, output_size, hidden_layers_size):

        # Start values for the NN
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers_size = np.array(hidden_layers_size)

        self.hidden_layers = self.hidden_layers_initialization()
    
    def xavier_initialization(self, shape, n):
        # Parameters: Shape of the data and N(amount of rows)
        # Xavier initialization for a layer
        init = []
        for _ in range(n):
            init.append(np.random.randn(shape[0], shape[1]) * np.sqrt(2.0/(shape[0])))

        return np.array(init).astype(np.float32)

    def hidden_layers_initialization(self):
        hidden_layers = []
        layer = self.hidden_layers_size[0]
        hidden_layers.append(self.xavier_initialization(self.input_size,layer))

        n = self.hidden_layers_size.shape[0]
        for i in range(1,n):
            current_layer = self.hidden_layers_size[i]
            hidden_layers.append(self.xavier_initialization(layer,current_layer))
            layer = current_layer

        return np.array(hidden_layers)



a = NN(input_size = 1, output_size = 1, hidden_layers_size = [2,3,4])
print(a.hidden_layers)