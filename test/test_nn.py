# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
import numpy as np
from nn import preprocess as pp
from itertools import repeat # For test_sample_seqs
import random # For test_sample_seqs
from nn import NeuralNetwork # for testing NN functions


# TODO: Write your test functions and associated docstrings below.

# Create a dummy NN to use in testing
arch = [{'input_dim': 64, 'output_dim': 32}, {'input_dim': 32, 'output_dim': 8}]
my_nn = NeuralNetwork(arch, 
                 lr=0.5,
                 seed=3,
                 batch_size=5,
                 epochs=5,
                 loss_function="sigmoid")


def test_forward():
    pass


def test_single_forward():
    pass


def test_single_backprop():
    pass


def test_predict():
    pass


def test_binary_cross_entropy():
    """
    Check that binary cross entropy loss is calculated correctly.
    """
    bce_error = round(my_nn._binary_cross_entropy(np.array([1,0]), np.array([0,1])))
    assert(bce_error == 12)


def test_binary_cross_entropy_backprop():
    pass


def test_mean_squared_error():
    """
    Check that MSE loss is calculated correctly.
    """
    assert(my_nn._mean_squared_error(np.array([0, 1]), np.array([1, 0])) == 1)


def test_mean_squared_error_backprop():
    pass


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

