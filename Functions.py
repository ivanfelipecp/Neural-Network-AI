import numpy as np

class Functions():

    # This is softmax
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        """ ivan
        e_x = np.exp(x - np.max(x))
        return e_x / np.array([np.sum(e_x, axis=1)]).T # only difference
        """

        e_x = np.exp(x - np.max(x))
        e_x += np.finfo(float).eps
        suma = np.sum(e_x, 1)
        return  e_x / suma[:,None]

    def relu(self, x):
        return np.maximum(x,0)

    def relu_prime(self, x):
        return np.heaviside(x,0)

    def predict(self,x):
        return self.relu(x)

    def cross_entropy(self, predict):
        loss = -np.log(predict)
        #print("\nLoss: ",loss)
        return np.sum(loss) / predict.shape[0]

    def cross_entropy_prime(self, predict, y):
        predict[range(y.size),y] =- 1
        return predict

    def one_hot_vector(self, output_size, y):
        a = np.array(y)
        b = np.zeros((a.shape[0],output_size))
        b[np.arange(a.shape[0]),a] = 1
        return b

    def dropout(self, X, dropout):
        return X * ((np.random.rand(*X.shape) < dropout) / dropout)