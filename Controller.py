from NN import NN
import tickle
import cv2
#from MNIST import MNIST
class Controller():
    def __init__(self):
    	self.nn = None#NN()
    
    def set_file(self,file):
        #print("seteo",file)
        self.nn = NN()
        self.nn.config(tickle.load(file))
        return "Archivo seteado con éxito"
    def classify(self,image):
    	msg = False
    	if self.nn:
    		img = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    		msg = str(self.nn.classify_image([img.flatten()]))
    	return msg
    def get_input_size(self):
    	return "Input size: "+ str(self.nn.input_size)
    def get_output_size(self):
    	return "Output size: "+str(self.nn.output_size)
    def get_hidden_layers(self):
    	return "Hidden layers(Ws): "+str(self.nn.hidden_layers_size)
    def get_learning_rate(self):
    	return "Learning rate: "+str(self.nn.learning_rate)
    def get_dropout(self):
    	return "Dropout: "+str(self.nn.dropout)
