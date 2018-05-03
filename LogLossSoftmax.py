import numpy as np

class LogLossSoftmax():

    # This is softmax
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        #print(x)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference

    def relu(self, x):
        return np.maximum(x,0)

    def relu_prime(self, x):
        return np.heaviside(x,0)

    def predict(self,x):
        return self.relu(x)

    def loss(self, predict):
        one_hot_vector = np.identity(predict.shape[0])
        return - (one_hot_vector * np.log(predict))

    def gradient(self, predict):
        return predict - 1