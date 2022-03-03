# from nn import preprocess as pp
# from itertools import repeat
# import random
# from nn import NeuralNetwork
# import numpy as np

import numpy as np
from nn import io
from nn import preprocess as pp
from nn import NeuralNetwork # for testing NN functions

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

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

	
	# arch = [{'input_dim': 64, 'output_dim': 32}, {'input_dim': 32, 'output_dim': 8}]
	# my_nn = NeuralNetwork(arch, 
	# 			 lr=0.5,
 #                 seed=3,
 #                 batch_size=5,
 #                 epochs=5,
 #                 loss_function="BCE")

	# Test _sigmoid
	#print(my_nn._sigmoid(np.array([0, 0, 0])))
	# Test _relu
	#print(my_nn._relu(np.array([0, -9, 10, 5.6])))
	# Test _binary_cross_entropy
	#print(my_nn._binary_cross_entropy(np.array([1,0]), np.array([0,1])))
	# Test _mean_squared_error
	#print(my_nn._mean_squared_error(np.array([0, 1]), np.array([1, 0])))

		# Create a dummy NN
	# arch = [{'input_dim': 4, 'output_dim': 2, 'activation': "relu"}, {'input_dim': 2, 'output_dim': 1, 'activation': "relu"}]
	# my_nn = NeuralNetwork(arch, 
	# 				 lr=0.5,
	# 				 seed=3,
	# 				 batch_size=1,
	# 				 epochs=1,
	# 				 loss_function="bce")

	# # Fix initial weights and biases
	# test_params = {}
	# test_params['W1'] = np.array([[1, 2, 3, 4],
	# 							 [3, 4, 5, 6]])
	# test_params['b1'] = np.array([[2],[2]])
	# test_params['W2'] = np.array([[2, 3]])
	# test_params['b2'] = np.array([[5]])
	# my_nn._set_params_for_test(test_params)

	# # Initialize toy dataset
	# test_X_train = np.array([[1, 2, 3, 4]])
	# test_Y_train = np.array([[1]])
	# test_X_val = np.array([[1, 2, 3, 4]])
	# test_Y_val = np.array([[1]])

	# # Fit NN
	# per_epoch_loss_train, per_epoch_loss_val = my_nn.fit(test_X_train, test_Y_train, test_X_val, test_Y_val)
	# #print(per_epoch_loss_train,per_epoch_loss_val)

	# # Make final predictions 
	# test_predict = my_nn.predict(test_X_train)
	# #print(test_predict)

	# Compare expected to actual predictions
	# expected_predict = np.array([[0]])
	# assert(per_epoch_loss_train[0] == 50176)
	# assert(np.array_equal(test_predict, expected_predict))


	# Use the 'read_text_file' function from io.py to read in the 137 positive Rap1 motif examples
	pos_seq_list = io.read_text_file('./data/rap1-lieb-positives.txt')

	# Use the 'read_fasta_file' function to read in all the negative examples from all 1kb upstream in yeast.
	neg_seq_list = io.read_fasta_file('./data/yeast-upstream-1k-negative.fa')

	# First, address class imbalance by sampling with replacement
	pos_labels = [1 for seq in pos_seq_list]
	neg_labels = [0 for seq in neg_seq_list]

	sample_size = len(pos_seq_list) + len(neg_seq_list)
	seqs, truth_labels = pp.sample_seqs(pos_seq_list+neg_seq_list, pos_labels+neg_labels, sample_size)

	# One hot encode sequences (NOTE: THIS MIGHT TAKE A WHILE TO RUN!)
	X = pp.one_hot_encode_seqs(seqs)
	y = np.expand_dims(np.asarray(truth_labels), axis=1)

	# Split into training and validation 
	# Place 1/3 of dataset into validation
	# X_train.shape = (2211, 4000)
	# y_train.shape = (2211, 1)
	# X_val.shape = (1089, 4000)
	# y_val.shape = (1089, 1)

	X_train, X_val, y_train, y_val = train_test_split(X, y , test_size=0.33, random_state=42)

	arch = [{'input_dim': 4000, 'output_dim': 2000, 'activation': "sigmoid"}, {'input_dim': 2000, 'output_dim': 1, 'activation': "sigmoid"}]
	my_nn = NeuralNetwork(arch, 
					 lr=0.01,
					 seed=15,
					 batch_size=10,
					 epochs=5,
					 loss_function="bce")

	# Train your neural network!
	per_epoch_loss_train, per_epoch_loss_val = my_nn.fit(X_train, y_train, X_val, X_val)

	print(per_epoch_loss_train)
	print(per_epoch_loss_val)





if __name__ == "__main__":
	main()