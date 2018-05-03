import numpy as np

class Functions():

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

    def cross_entropy(self, predict, y):
        one_hot_vector = self.one_hot_vector(np.int_(predict.shape[1]), np.int_(y))
        predict = np.transpose(predict)

        loss = - np.dot(one_hot_vector,np.log(predict))
        print(np.sum(loss))
        predict = np.transpose(predict)
        one_hot_vector = np.transpose(one_hot_vector)
        loss = - np.dot(one_hot_vector,np.log(predict))
        print(np.sum(loss))
        input("waiting....")
        return np.sum(loss)

    def gradient(self, predict):
        return predict - 1

    def one_hot_vector(self, output_size, y):
        a = np.array(y)
        b = np.zeros((a.shape[0],output_size))
        b[np.arange(a.shape[0]),a] = 1
        return b