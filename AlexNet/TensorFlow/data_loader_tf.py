import pickle
import numpy as np
import tensorflow as tf 


class CifarDataset():
	""" Load CIFAR dataset """

	def __init__(self, config):
		# Load config file 
		self.config = config 
		# Load data from data path 
		self.X_train, self.y_train, self.X_test, self.y_test = self.load_cifar10(
			config["data_loader"]["data_path"])


	def unpickle(self, file):
		with open(file, 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
		return dict 


	def load_cifar10(self, data_path):
		train_data = None
		train_labels = []
		test_data = None
		test_labels = None

		for i in range(1, 6):
			data_dict = self.unpickle(data_path + "data_batch_" + str(i))
			if (i == 1):
				train_data = data_dict[b'data']
			else:
				train_data = np.vstack((train_data, data_dict[b'data']))
			train_labels += data_dict[b'labels']

		test_data_dict = self.unpickle(data_path + "test_batch")
		test_data = test_data_dict[b'data']
		test_labels = test_data_dict[b'labels']

		train_data = train_data.reshape((50000, 3, 32, 32))
		train_data = np.rollaxis(train_data, 1, 4)
		train_labels = np.array(train_labels)

		test_data = test_data.reshape((10000, 3, 32, 32))
		test_data = np.rollaxis(test_data, 1, 4)
		test_labels = np.array(test_labels)

		return train_data, train_labels, test_data, test_labels		


