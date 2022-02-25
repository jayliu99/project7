# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
import numpy as np
from nn import preprocess as pp
from itertools import repeat # For test_sample_seqs
import random # For test_sample_seqs
from nn import NeuralNetwork # for testing NN functions


# TODO: Write your test functions and associated docstrings below.

# Create a dummy NN to use in testing
arch = [{'input_dim': 4, 'output_dim': 2, 'activation': "relu"}, {'input_dim': 2, 'output_dim': 1, 'activation': "relu"}]
my_nn = NeuralNetwork(arch, 
				 lr=0.5,
				 seed=3,
				 batch_size=5,
				 epochs=5,
				 loss_function="BCE")


def test_forward():
	"""
	Check that an entire forward pass performs as expected.
	"""

	# Set fixed input, weights, and biases
	test_params = {}
	test_params['W1'] = np.array([[1, 2, 3, 4],
								 [3, 4, 5, 6]])
	test_params['b1'] = np.array([[2],[2]])
	test_params['W2'] = np.array([[2, 3]])
	test_params['b2'] = np.array([[5]])

	my_nn._set_params_for_test(test_params)

	test_A_prev = np.array([[1, 2, 3, 4]])

	# Run forward pass
	output, cache = my_nn.forward(test_A_prev)

	# Define expected output
	expected_output = np.array([[225]]) 
	expected_cache = {}
	expected_cache['Z1'] = np.array([[32, 52]])
	expected_cache['A1'] = np.array([[32, 52]])
	expected_cache['Z2'] = np.array([[225]]) 
	expected_cache['A2'] = np.array([[225]]) 

	# Compare expected to actual output
	assert(np.array_equal(output, expected_output))
	assert(cache.keys() == expected_cache.keys())
	assert(np.array_equal(cache['Z1'], expected_cache['Z1']))
	assert(np.array_equal(cache['A1'], expected_cache['A1']))
	assert(np.array_equal(cache['Z2'], expected_cache['Z2']))
	assert(np.array_equal(cache['A2'], expected_cache['A2']))


def test_single_forward():
	"""
	Check that a single forward pass performs as expected.
	"""

	# Set fixed input, weights, and biases
	test_A_prev = np.array([[1, 2, 3, 4],
							 [1, 2, 3, 4]])
	test_W_curr = np.array([[1, 2, 3, 4],
							[3, 4, 5, 6]])
	test_b_curr = np.array([[2], [2]])

	# Run single forward pass (with relu activation)
	test_A, test_Z = my_nn._single_forward(test_W_curr, test_b_curr, test_A_prev, "relu")
	expected_A = np.array([[32, 52], [32, 52]])

	# Evaluate if results were as expected
	assert(np.array_equal(test_A, expected_A))

	# Run single forward pass (with sigmoid activation)
	test_A, test_Z = my_nn._single_forward(test_W_curr, test_b_curr, test_A_prev, "sigmoid")
	expected_A = np.array([[1, 1], [1, 1]])

	# Evaluate if results were as expected
	assert(np.array_equal(np.round(test_A), expected_A))


def test_single_backprop():
	"""
	Check that a single backward pass performs as expected.
	"""

	# Set fixed input, weights, and biases
	test_W_curr = np.array([[1, 2, 3, 4],
							[3, 4, 5, 6]])
	test_b_curr = np.array([[2], [2]])
	test_Z_curr = np.array([[32, 52], [32, 52]])
	test_A_prev = np.array([[1, 2, 3, 4],
							 [1, 2, 3, 4]])
	test_dA_curr = np.array([[1, 3], [2, 4]])
	test_act_curr = "relu"

	# Run single backward pass (with relu activation)
	test_dA_prev, test_dW_curr, test_db_curr = my_nn._single_backprop(test_W_curr, 
											test_b_curr, 
											test_Z_curr, 
											test_A_prev, 
											test_dA_curr,
											test_act_curr)
	expected_dA_prev = np.array([[10, 14, 18, 22], [14, 20, 26, 32]])
	expected_dW_curr = np.array([[3, 6, 9, 12], [7, 14, 21, 28]])
	expected_db_curr = np.array([[3], [7]])


	# Evaluate if results were as expected
	assert(np.array_equal(test_dA_prev, expected_dA_prev))
	assert(np.array_equal(test_dW_curr, expected_dW_curr))
	assert(np.array_equal(test_db_curr, expected_db_curr))



def test_predict():
	pass


def test_binary_cross_entropy():
	"""
	Check that binary cross entropy loss is calculated correctly.
	"""
	bce_error = round(my_nn._binary_cross_entropy(np.array([1,0]), np.array([0,1])))
	assert(bce_error == 12)


def test_binary_cross_entropy_backprop():
	"""
	Check that the derivative of MSE loss with respect to y_hat is calculated correctly.
	"""
	epsilon = 1e-5
	expected = np.array([[1/12], [3/4]])
	test = my_nn._binary_cross_entropy_backprop(np.array([[4], [5]]), np.array([[3], [2]]))
	assert(np.sum(expected-test) < epsilon)


def test_mean_squared_error():
	"""
	Check that MSE loss is calculated correctly.
	"""
	assert(my_nn._mean_squared_error(np.array([0, 1]), np.array([1, 0])) == 1)


def test_mean_squared_error_backprop():
	"""
	Check that the derivative of MSE loss with respect to y_hat is calculated correctly.
	"""
	expected = np.array([[-4], [-4]])
	test = my_nn._mean_squared_error_backprop(np.array([[4], [5]]), np.array([[0], [1]]))
	assert(np.array_equal(expected, test))

def test_one_hot_encode():
	"""
	Check that one-hot enocding translation is correct for all 
	possible charaters, and that function can handle samples with
	differently sized sequences.
	"""

	test = pp.one_hot_encode_seqs(["AGA", "A", "CG"])
	truth = np.array([[1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.],
					  [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
					  [0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])
	assert(np.array_equal(test, truth))


def test_sample_seqs():
	"""
	Check that a balanced sample set is generated from an imbalanced
	dataset.
	"""

	# Generate imbalanced toy dataset
	toy_data = [("A", True), ("B", False)]
	toy_data_pos = [x for item in toy_data for x in repeat(item, 2) if item[0]=="A"]
	toy_data_neg = [x for item in toy_data for x in repeat(item, 10) if item[0]=="B"]
	toy_data = toy_data_pos + toy_data_neg
	random.shuffle(toy_data)
	seqs = [item[0] for item in toy_data]
	labels = [item[1] for item in toy_data]

	# Assert that returned sample is balanced
	sample_seqs, sample_labels = pp.sample_seqs(seqs, labels, 20)
	num_pos_labels = sample_labels.count(True)
	num_neg_labels = sample_labels.count(False)

	assert( -1 <= num_pos_labels-num_neg_labels <= 1)

