import json 
import logging 
from data_loader_tf import CifarDataset 
from model import AlexNet 

from matplotlib import pyplot as plt
import pdb

import tensorflow as tf

from model import model_fn

def main():
	with open("config.json", "r") as f:
		config = json.load(f)

	# Load Cifar data 
	data = CifarDataset(config)

	# x, y = train_input_fn(config, data)

	# pdb.set_trace()

	model = tf.estimator.Estimator(model_fn=model_fn, 
		params=config, 
		model_dir="./saved/")

	# pdb.set_trace()

	## Training 
	train_input_fn_wrapper_estimator = train_input_fn_wrapper(data=data, config=config, train=True)

	# pdb.set_trace()

	model.train(input_fn=train_input_fn_wrapper_estimator, 
		steps=config["trainer"]["epochs"] * config["trainer"]["batch_size"])


	## Evaluation 
	test_input_fn_wrapper_estimator = test_input_fn_wrapper(data=data, config=config, train=False)
	result = model.evaluate(input_fn=test_input_fn_wrapper_estimator)
	print("result: {:.3f}".format(result))


def preprocess_data(data, config, train):
	# Rescale images
	data = data.map(lambda x, y: (tf.div(tf.cast(x, tf.float32), 255.0), y))	
	# Resize images
	data = data.map(lambda x, y: (tf.image.resize_images(x, [256, 256]), y))
	# One-hot the output
	data = data.map(lambda x, y: (x, tf.one_hot(y, 10)))

	if train: 
		# IF training, read a buffer and randomly shuffle it
		data = data.shuffle(buffer_size=config["trainer"]["buffer_size"])
		# Allow infinite reading of the data in training 
		num_repeat = None
	else:
		# if testing then don't shuffle the data 
		num_repeat = 1

	# Repeat the dataset the given number of times 
	data = data.repeat(num_repeat)

	if train: 
		data = data.batch(config["trainer"]["batch_size"])  		
	else:
		data = data.batch(config["evaluation"]["batch_size"])

	# Create an iterator for the dataset and the above modifications
	iterator = data.make_one_shot_iterator()
	# Get the next batch of images and labels
	images_batch, labels_batch = iterator.get_next()

	# The input-function must return a dict wrapping the images
	x = {'image': images_batch}
	y = labels_batch 

	return (x, y)	


def train_input_fn_wrapper(config, data):
	def train_input_fn():
		# Extract train data
		X_train, y_train = data.X_train, data.y_train
		# Create tf.data.Dataset instance from numpy array
		train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

		# preprocess data
		x, y = preprocess_data(train_dataset, config, train=True)

		# # # Rescale images
		# # train_dataset = train_dataset.map(lambda x, y: (tf.div(tf.cast(x, tf.float32), 255.0), y))	
		# # # Resize images 
		# # train_dataset = train_dataset.map(lambda x, y: (tf.image.resize_images(x, [256, 256]), y))

		# # TODO : Add image augmentation functions

		# # Read a buffer and randomly shuffle it
		# train_dataset = train_dataset.shuffle(buffer_size=config["trainer"]["buffer_size"])
		# # Allow infinite reading of the data in training 
		# num_repeat = None 
		# train_dataset = train_dataset.repeat(count=num_repeat)
		# # Get a batch of the given size
		# train_dataset = train_dataset.batch(config["trainer"]["batch_size"])
		# # Create an iterator for the dataset and the above modifications
		# iterator = train_dataset.make_one_shot_iterator()
		# # Get the next batch of images and labels
		# images_batch, labels_batch = iterator.get_next()

		# # The input-function must return a dict wrapping the images
		# x = {'image': images_batch}
		# y = labels_batch 

		return (x, y)		

	""" Wrap the train_input_fn because the estimator function does not accept
	any input arguments """
	return train_input_fn # return function call 



def test_input_fn_wrapper(config, data):
	def test_input_fn():
		# Extract train data
		X_test, y_test = data.X_test, data.y_test
		# Create tf.data.Dataset instance from numpy array
		test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

		# preprocess data
		x, y = preprocess_data(test_dataset, data, train=False)

		return x, y

	""" Wrap the test_input_fn because the estimator function does not accept
	any input arguments """
	return test_input_fn # return function call


# def model_fn()

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')

	main()