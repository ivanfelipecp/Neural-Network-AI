from NN import NN
from MNIST import MNIST
class Controller():
    def __init__(self):
        self.nn = NN()

    def set_file(self,file):
        print("seteo",file)
        return "Archivo seteado con éxito"

    def classify(self,image):
        print("clasifico",image)
        return "mensaje del clasificador"
