import pickle
import numpy as np
import cv2
from torch.utils import data


class CifarDataLoader(data.Dataset):
	def __init__(self, config, data_X, data_y, transform=None):
		self.config = config
		self.images, self.labels = data_X, data_y 
		self.transform = transform

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		# print("self.images[idx].shape: ", self.images[idx].shape)
		image = self.images[idx]
		label = self.labels[idx]

		sample = {'image': image, 'label': label}

		if self.transform:
			sample = self.transform(sample)

		return sample

class CifarDataset(object):
	""" Load CIFAR dataset """
	def __init__(self, config):
		# Load config file
		self.config = config
		# Load data from data_path
		self.X_train, self.y_train, self.X_test, self.y_test = self.load_cifar10(self.config["data_loader"]["data_path"])
		# shuffle data
		self.shuffle()
		# One-hot encode the labels
		self.y_train, self.y_test = self.one_hot_labels(self.y_train), self.one_hot_labels(self.y_test)
		# Split train into train/validation
		# self.X_train, self.X_valid, self.y_train, self.y_valid = self.split(self.X_train, self.y_train)
		if config["validation"]["split"]: 
			self.X_train, self.X_valid, self.y_train, self.y_valid = self.split()

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


	def one_hot_labels(self, label_data, num_classes=10):
		"""
		One hot encode the labels
		"""
		label_data = np.eye(num_classes)[label_data.reshape(-1)]

		return label_data

	# def split(self, train_data, train_labels):
	# 	""" Split train_data into train/validation set """
	# 	return train_test_split(train_data, train_labels, test_size=0.1, random_state=42)

	def split(self):
		""" Split train_data into train/validation set """
		validation_split = self.config["validation"]["validation_split"]
		train_elem = int(self.X_train.shape[0] * (1 - validation_split))
		X_train, y_train = self.X_train[:train_elem], self.y_train[:train_elem]
		X_valid, y_valid = self.X_train[train_elem:], self.y_train[train_elem:] 

		return X_train, X_valid, y_train, y_valid 

	def shuffle(self):
		""" Shuffle the data """ 
		if self.config["data_loader"]["shuffle"]:
			indices = np.arange(self.X_train.shape[0])
			np.random.shuffle(indices)
			self.X_train, self.y_train = self.X_train[indices], self.y_train[indices]


