import numpy as np

class LogLossSoftmax():

    # This is softmax
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / np.array([np.sum(e_x, axis=1)]).T # only difference

    def relu(self, x):
        return x * (x > 0)

    def relu_prime(self, x):
        return 1. * (x > 0)#np.heaviside(x,0)

    def predict(self,x):
        return self.relu(x)

    def loss(self, predict):
        one_hot_vector = np.identity(predict.shape[0])
        return - (one_hot_vector * np.log(predict))

    def gradient(self, predict):
        return predict - 1