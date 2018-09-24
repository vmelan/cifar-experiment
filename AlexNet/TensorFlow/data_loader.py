import pickle
import numpy as np
import cv2
import random


class CifarDataset():
	""" Load CIFAR dataset """
	def __init__(self, config):
		# Load config file
		self.config = config
		# Load data from data_path
		self.X_train, self.y_train, self.X_test, self.y_test = self.load_cifar10(
			config["data_loader"]["data_path"])
		# Shuffle data
		self.shuffle()
		# Scale inputs 
		self.X_train, self.X_test = self.scale_data(self.X_train), self.scale_data(self.X_test)

		# One-hot encode the labels
		self.y_train, self.y_test = self.one_hot_labels(self.y_train), self.one_hot_labels(self.y_test)
		# Split train into train/validation
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


	def scale_data(self, data):
		""" 
		Scale the row pixel intensities to the range [0, 1]
		"""

		data = data.astype(np.float32) / 255.0
		return data

	def convert_grayscale(self, data_image):
		"""
		Convert image to grayscale
		"""
		output = np.zeros((data_image.shape[:-1]))
		for i, image in enumerate(data_image):
			gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			output[i] = gray_image

		return output.reshape(-1, 32, 32, 1)




	def one_hot_labels(self, label_data, num_classes=10):
		"""
		One hot encode the labels
		"""
		label_data = np.eye(num_classes)[label_data.reshape(-1)]

		return label_data


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





class CifarDataLoader(CifarDataset):
	""" CIFAR data with augmentation """

	def __init__(self, config):
		super(CifarDataLoader, self).__init__(config)


	def _resize(self, data_image, new_width=256, new_height=256):
		""" Resize image """ 
		output = np.zeros((data_image.shape[0], new_height, new_width, 3))
		for i, image in enumerate(data_image):
			resized_img = cv2.resize(image, (new_width, new_height))
			output[i] = resized_img

		return output


	def _random_crop(self, data_image, new_width=227, new_height=227):
		""" Random Crop of a given image """
		height, width = data_image.shape[1:3]
		
		output = np.zeros((data_image.shape[0], new_height, new_width, 3))
		for i, image in enumerate(data_image):
			top = np.random.randint(0, height - new_height)
			left = np.random.randint(0, width - new_width)
			output[i] = image[top:top + new_height, left:left + new_width]

		return output


	def _random_horizontal_flip(self, data_image, p=0.5):
		""" Random horizontally flip of a given image """
		
		batch_size, height, width, channels = data_image.shape
		output = np.zeros((batch_size, height, width, channels))
		for i, image in enumerate(data_image):
			if random.random() < p:
				# print("image", i, "flipping")
				output[i] = image[:, ::-1, :]
			else:
				output[i] = image

		return output


	def get_next_batch(self):
		""" Yield batches for training """

		idx = np.random.choice(self.X_train.shape[0], size=self.config["trainer"]["batch_size"])
		batch_X, batch_y = self.X_train[idx], self.y_train[idx]
		# Resize image
		batch_X = self._resize(batch_X, new_width=256, new_height=256)
		# Random crop 
		batch_X = self._random_crop(batch_X, new_width=227, new_height=227)
		# Random horizontal flip 
		batch_X = self._random_horizontal_flip(batch_X, p=0.5)

		yield (batch_X, batch_y)