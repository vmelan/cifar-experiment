import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

import matplotlib.image as mpimg


import torch
from torch.utils import data


class CifarDataset(data.Dataset):
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


class ToTensor(object):
	""" Convert ndarrays in sample to Tensors. """

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		# Swap color axis because numpy image : H x W x C and torch image : C x H x W
		image = image.transpose((2, 0, 1))

		return {'image': torch.from_numpy(image), 
				'label': torch.from_numpy(label)}


class Normalize(object):
	""" Normalize the color range of an image to [0, 1] """

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		# scale color range from [0, 255] to [0, 1]
		image = image / 255.0

		return {'image': image, 'label': label}


class ToGrayscale(object):
	""" Conver a color image to grayscale """

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray_image = gray_image.reshape(32, 32, 1)

		return {'image': gray_image, 'label': label}


class CifarDataLoader(object):
	""" Load CIFAR dataset """
	def __init__(self, config):
		# Load config file
		self.config = config
		# Load data from data_path
		self.X_train, self.y_train, self.X_test, self.y_test = self.load_cifar10(config["data_path"])
		# Scale inputs 
		# self.X_train, self.X_test = self.scale_data(self.X_train), self.scale_data(self.X_test)
		# Convert inputs to grayscale
		# self.X_train, self.X_test = self.convert_grayscale(self.X_train), self.convert_grayscale(self.X_test)
		# One-hot encode the labels
		self.y_train, self.y_test = self.one_hot_labels(self.y_train), self.one_hot_labels(self.y_test)
		# Split train into train/validation
		self.X_train, self.X_valid, self.y_train, self.y_valid = self.split(self.X_train, self.y_train)

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

	def split(self, train_data, train_labels):
		""" Split train_data into train/validation set """
		return train_test_split(train_data, train_labels, test_size=0.1, random_state=42)

	def next_batch(self):
		""" Yield batches for training """

		while True:
			idx = np.random.choice(self.X_train.shape[0], size=self.config["batch_size"])
			yield (self.X_train[idx], self.y_train[idx])

