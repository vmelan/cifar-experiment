import json 
import logging 
from data_loader_tf import CifarDataset 
from model import AlexNet 

from matplotlib import pyplot as plt
import pdb

import tensorflow as tf

def main():
	with open("config.json", "r") as f:
		config = json.load(f)

	# Load Cifar data 
	data = CifarDataset(config)

	(X_train, y_train) = data.X_train, data.y_train 
	(X_test, y_test) = data.X_test, data.y_test

	train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)

	# X_train = tf.data.Dataset.from_tensor_slices(X_train)
	# y_train = tf.data.Dataset.from_tensor_slices(y_train)
	# Rescale images
	train_dataset = train_dataset.map(lambda x, y: (tf.div(tf.cast(x, tf.float32), 255.0), y))
	# Resize images 
	train_dataset = train.dataset.map()
	# One hot 
	train_dataset = train_dataset.map(lambda x, y: (x, tf.one_hot(y, 10)))

	pdb.set_trace()

	# train_dataset = train_dataset.batch(32)

	pdb.set_trace()

	# Test
	# batch_X, batch_y = next(data.get_next_batch())

	# # Create AlexNet model 
	# net = AlexNet(config)


if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')

	main()