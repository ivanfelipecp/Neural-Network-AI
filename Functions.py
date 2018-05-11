import numpy as np

class Functions():

    # This is softmax
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        """
        e_x = np.exp(x - np.max(x))
        return e_x / np.array([np.sum(e_x, axis=1)]).T # only difference
        """

        """e_x = np.exp(x - np.max(x))
        e_x += np.finfo(float).eps
        suma = np.sum(e_x, 1)
        return  e_x / suma[:,None]"""

        exp_scores = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        #exps = np.exp(x - np.max(x))
        #return exps / np.sum(exps)

        """max_per_row = np.reshape(np.max(x, axis=1), (x.shape[0], 1))
        exp_scores = np.exp(x - max_per_row)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs"""

    def activation(self, x):
        return self.relu(x)
        #return self.sigmoid(x)

    def activation_prime(self, x):
        return self.relu_prime(x)
        #return self.sigmoid_prime(x)

    def sigmoid(self, x):
        # activation function 
        return 1/(1+np.exp(-x))

    def sigmoid_prime(self, x):
        # derivative of sigmoid
        return x * (1 - x)

    def relu(self, x):
        ##return np.maximum(x,0,x)
        #return np.maximum(0,x)
        #return x * (x > 0)
        x[x<0] = 0
        return x

    def relu_prime(self, x):
        #return np.heaviside(x,0)
        #return 1. * (x > 0)
        x[x>0] = 1
        return x

    def safe_ln(self, x, minval=0.00001, maxval = 0.1):
        return np.log(np.clip(x,minval,maxval))

    def cross_entropy(self, p, y):
        # Loss
        y = self.one_hot_encode(p.shape, y)
        return np.mean(np.sum(np.nan_to_num(-y * self.safe_ln(p) - (1 - y) * self.safe_ln(1 - p)), axis = 1))
        #loss = np.mean(np.sum(np.nan_to_num(-y * np.log(predict) - (1 - y) * np.log(1 - predict)), axis = 1))
        #return np.sum(loss) / predict.shape[0]

        #return - np.multiply(y, np.log(predict))

        """i = range(p.shape[0])
        L_i = -np.log(p[i,y.astype(int)[i]])
        loss = 1/L_i.shape[0] * np.sum(L_i)
        return loss"""

        """m = Y.shape[0]
        log_likelihood = -np.log(predict[range(m),Y])
        loss = np.sum(log_likelihood) / m

        return np.sum(loss) / predict.shape[0]"""

    def cross_entropy_prime(self, p, y):
        #p[range(y.size),y] -= 1
        #return p
        #y = self.one_hot_encode(p.shape, y)
        return p - y

        """m = y.shape[0]
        #grad = softmax(predict)
        predict[range(m),y] -= 1
        predict = predict/m
        return predict"""

    def one_hot_encode(self, shape, Y):
        y = np.zeros(shape)
        y[np.arange(Y.shape[0]), Y] = 1
        return y.astype(np.int_)

    def dropout(self, X, dropout):
        #return (np.random.rand(*X.shape) < dropout) / dropout
        return np.random.binomial(size=X.shape[1], n=1, p=dropout)/dropout

    def make_y(self, shape, y):
        array = []
        n = len(y)
        for i in range(n):
            b = np.zeros(shape[1])
            b[y[i]] = y[i]
            array.append(b)
        return np.array(array)

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