from nn import preprocess as pp
from itertools import repeat
import random
from nn import NeuralNetwork
import numpy as np

def main():

	# Test pp.one_hot_encode_seqs
	#print(pp.one_hot_encode_seqs(["AGA", "A"]))

	# Test pp.sample_seqs
	# Generate toy dataset
	# toy_data = [("A", True), ("B", False)]
	# toy_data_pos = [x for item in toy_data for x in repeat(item, 2) if item[0]=="A"]
	# toy_data_neg = [x for item in toy_data for x in repeat(item, 10) if item[0]=="B"]
	# toy_data = toy_data_pos + toy_data_neg
	# random.shuffle(toy_data)
	# seqs = [item[0] for item in toy_data]
	# labels = [item[1] for item in toy_data]
	# print(pp.sample_seqs(seqs, labels, 20))

	
	arch = [{'input_dim': 64, 'output_dim': 32}, {'input_dim': 32, 'output_dim': 8}]
	my_nn = NeuralNetwork(arch, 
				 lr=0.5,
                 seed=3,
                 batch_size=5,
                 epochs=5,
                 loss_function="BCE")

	# Test _sigmoid
	#print(my_nn._sigmoid(np.array([0, 0, 0])))
	# Test _relu
	#print(my_nn._relu(np.array([0, -9, 10, 5.6])))
	# Test _binary_cross_entropy
	#print(my_nn._binary_cross_entropy(np.array([1,0]), np.array([0,1])))
	# Test _mean_squared_error
	#print(my_nn._mean_squared_error(np.array([0, 1]), np.array([1, 0])))




if __name__ == "__main__":
	main()