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




	x, y = train_input_fn(config, data)

	# pdb.set_trace()

	model = tf.estimator.Estimator(model_fn=model_fn, 
		params=config, 
		model_dir="./saved/")

	# pdb.set_trace()

	## Training 
	# model.train(input_fn=train_input_fn, 
	# 	steps=config["trainer"]["epochs"] * config["trainer"]["batch_size"])

	model.train(input_fn=train_input_fn_wrapper, 
		steps=config["trainer"]["epochs"] * config["trainer"]["batch_size"])

	# Test
	# batch_X, batch_y = next(data.get_next_batch())

	# # Create AlexNet model 
	# net = AlexNet(config)



def train_input_fn(config, data):
	# Extract train data
	X_train, y_train = data.X_train, data.y_train
	# Create tf.data.Dataset instance from numpy array
	train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
	# Rescale images
	train_dataset = train_dataset.map(lambda x, y: (tf.div(tf.cast(x, tf.float32), 255.0), y))	
	# Resize images 
	train_dataset = train_dataset.map(lambda x, y: (tf.image.resize_images(x, [256, 256]), y))

	# TODO : Add image augmentation functions
	
	# One hot 
	train_dataset = train_dataset.map(lambda x, y: (x, tf.one_hot(y, 10)))
	# Read a buffer and randomly shuffle it
	train_dataset = train_dataset.shuffle(buffer_size=config["trainer"]["buffer_size"])
	# Allow infinite reading of the data in training 
	num_repeat = None 
	train_dataset = train_dataset.repeat(count=num_repeat)
	# Get a batch of the given size
	train_dataset = train_dataset.batch(config["trainer"]["batch_size"])
	# Create an iterator for the dataset and the above modifications
	iterator = train_dataset.make_one_shot_iterator()
	# Get the next batch of images and labels
	images_batch, labels_batch = iterator.get_next()

	# The input-function must return a dict wrapping the images
	x = {'image': images_batch}
	y = labels_batch 

	return (x, y)



with open("config.json", "r") as f:
	config = json.load(f)

# Load Cifar data 
data = CifarDataset(config)

def train_input_fn_wrapper():
	""" Wrap the train_input_fn because the estimator function does not accept
	any input arguments """
	return train_input_fn(config=config, data=data)



# def model_fn()

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')

	main()