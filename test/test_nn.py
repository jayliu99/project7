# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
import numpy as np
from nn import preprocess as pp
from itertools import repeat # For test_sample_seqs
import random # For test_sample_seqs


# TODO: Write your test functions and associated docstrings below.

def test_forward():
    pass


def test_single_forward():
    pass


def test_single_backprop():
    pass


def test_predict():
    pass


def test_binary_cross_entropy():
    pass


def test_binary_cross_entropy_backprop():
    pass


def test_mean_squared_error():
    pass


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

    # Assert that sample is balanced
    sample_seqs, sample_labels = pp.sample_seqs(seqs, labels, 20)
    num_pos_labels = sample_labels.count(True)
    num_neg_labels = sample_labels.count(False)

    assert( -1 <= num_pos_labels-num_neg_labels <= 1)

