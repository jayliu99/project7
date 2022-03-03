# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import random # For sample_seqs


# Defining preprocessing functions
def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
	"""
	This function generates a flattened one hot encoding of a list of nucleic acid sequences
	for use as input into a fully connected neural net.

	Args:
		seq_arr: List[str]
			List of sequences to encode.

	Returns:
		encodings: ArrayLike
			Array of encoded sequences, with each encoding 4x as long as the input sequence
			length due to the one hot encoding scheme for nucleic acids.

			For example, if we encode 
				A -> [1, 0, 0, 0]
				T -> [0, 1, 0, 0]
				C -> [0, 0, 1, 0]
				G -> [0, 0, 0, 1]
			Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
	"""
	# Figure out dimensions of one-hot encoding array
	longest_seq = max(seq_arr, key=len)
	cols = 4 * len(longest_seq)
	rows = len(seq_arr)

	# Create an array to store one-hot encodings
	encodings = np.zeros((rows, cols))

	# Translate each sequence into a one-hot encoding, and append to the encodings array
	r_index = 0
	for seq in seq_arr:
		one_hot = []
		one_hot[:0] = seq # Transform string seq to a list, character-wise split
		
		A = [1, 0, 0, 0]
		T = [0, 1, 0, 0]
		C = [0, 0, 1, 0]
		G = [0, 0, 0, 1]

		# Translate A
		one_hot = [A if na == 'A' else [na] for na in one_hot]
		one_hot = [v for vals in one_hot for v in vals]	# Flatten list

		# Translate T
		one_hot = [T if na == 'T' else [na] for na in one_hot]
		one_hot = [v for vals in one_hot for v in vals] # Flatten list

		# Translate C
		one_hot = [C if na == 'C' else [na] for na in one_hot]
		one_hot = [v for vals in one_hot for v in vals] # Flatten list

		# Translate G
		one_hot = [G if na == 'G' else [na] for na in one_hot]
		one_hot = [v for vals in one_hot for v in vals] # Flatten list

		# Append new encoding to array of all encodings
		encodings[r_index, :len(one_hot)] = one_hot 
		r_index+=1

	return encodings




def sample_seqs(seqs: List[str], labels: List[bool], size:int) -> Tuple[List[str], List[bool]]:
	"""
	This function should sample your sequences to account for class imbalance. 
	Consider this as a sampling scheme with replacement.
	
	Args:
		seqs: List[str]
			List of all sequences.
		labels: List[bool]
			List of positive/negative labels
		size: int
			Size of sample to be returned.

	Returns:
		sampled_seqs: List[str]
			List of sampled sequences which reflect a balanced class size
		sampled_labels: List[bool]
			List of labels for the sampled sequences
	"""
	# Set seed 
	random.seed = 25
	
	# Segregate postive and negative data
	labeled_seq = list(zip(seqs, labels))
	positive_ex = [item[0] for item in labeled_seq if item[1]==True]
	negative_ex = [item[0] for item in labeled_seq if item[1]==False]

	# Sample with replacement
	sample_size = int(size/2)

	positive_sample = random.choices(positive_ex, k=sample_size) 
	negative_sample = random.choices(negative_ex, k=sample_size) 

	# Recombine samples
	positive_sample = [(ps, True) for ps in positive_sample]
	negative_sample = [(ns, False) for ns in negative_sample]
	sample = positive_sample + negative_sample

	# If number of samples to return is odd, randomly grab one more sample
	if sample_size%2 == 1:
		sample.append(random.sample(labeled_seq, k=1))

	# Shuffle samples
	random.shuffle(sample)

	# Return split list of seqs and labels
	sequences = [item[0] for item in sample]
	truth_labels = [item[1] for item in sample]
	return (sequences, truth_labels)






