from nn import preprocess as pp
from itertools import repeat
import random

def main():

	# Test pp.one_hot_encode_seqs
	#print(pp.one_hot_encode_seqs(["AGA", "A"]))

	# Test pp.sample_seqs
	# Generate toy dataset
	toy_data = [("A", True), ("B", False)]
	toy_data_pos = [x for item in toy_data for x in repeat(item, 2) if item[0]=="A"]
	toy_data_neg = [x for item in toy_data for x in repeat(item, 10) if item[0]=="B"]
	toy_data = toy_data_pos + toy_data_neg
	random.shuffle(toy_data)
	seqs = [item[0] for item in toy_data]
	labels = [item[1] for item in toy_data]

	print(pp.sample_seqs(seqs, labels, 20))


if __name__ == "__main__":
	main()