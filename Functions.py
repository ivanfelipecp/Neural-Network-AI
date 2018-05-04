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

    def cross_entropy(self, predict, Y):
        y = self.one_hot_encode(predict.shape, Y)
        loss = np.mean(np.sum(np.nan_to_num(-y * np.log(predict) - (1 - y) * np.log(1 - predict)), axis = 1))
        #print("\nLoss: ",loss)
        return np.sum(loss) / predict.shape[0]

    def cross_entropy_prime(self, predict, y):
        predict[range(y.size),y] =- 1
        return predict

    def one_hot_encode(self, shape, Y):
        y = np.zeros(shape)
        y[np.arange(Y.shape[0]), Y] = 1
        return y

    def dropout(self, X, dropout):
        return X * ((np.random.rand(*X.shape) < dropout) / dropout)

    
    def get_random_elements(self, batch_size, x, y):
        r_X = []
        r_Y = []
        x = list(x)
        y = list(y)
        for i in range(batch_size):
            ind = np.random.randint(0,len(x))
            r_X.append(x[ind])
            r_Y.append(y[ind])
            del x[ind]
            del y[ind]
        
        return np.array(r_X), np.array(r_Y), np.array(x), np.array(y)