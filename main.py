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
	arch = [{'input_dim': 4, 'output_dim': 2, 'activation': "relu"}, {'input_dim': 2, 'output_dim': 1, 'activation': "relu"}]
	my_nn = NeuralNetwork(arch, 
					 lr=0.5,
					 seed=3,
					 batch_size=1,
					 epochs=1,
					 loss_function="mse")

	# Fix initial weights and biases
	test_params = {}
	test_params['W1'] = np.array([[1, 2, 3, 4],
								 [3, 4, 5, 6]])
	test_params['b1'] = np.array([[2],[2]])
	test_params['W2'] = np.array([[2, 3]])
	test_params['b2'] = np.array([[5]])
	my_nn._set_params_for_test(test_params)

	# Initialize toy dataset
	test_X_train = np.array([[1, 2, 3, 4]])
	test_Y_train = np.array([[1]])
	test_X_val = np.array([[1, 2, 3, 4]])
	test_Y_val = np.array([[1]])

	# Fit NN
	per_epoch_loss_train, per_epoch_loss_val = my_nn.fit(test_X_train, test_Y_train, test_X_val, test_Y_val)
	#print(per_epoch_loss_train,per_epoch_loss_val)

	# Make final predictions 
	test_predict = my_nn.predict(test_X_train)
	#print(test_predict)

	# Compare expected to actual predictions
	expected_predict = np.array([[0]])
	assert(per_epoch_loss_train[0] == 50176)
	assert(np.array_equal(test_predict, expected_predict))




if __name__ == "__main__":
	main()