import _pickle as pickle


def save(nn_weights, name):
	pickle.dump(nn_weights, open(name, "wb"))

def load(name):
	return pickle.load(open(name, "rb"))