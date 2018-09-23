import torch
import numpy as np
import cv2

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


class Resize(object):
	""" Resize image to given size 

	Args: 
		output_size (tuple or int): Desired output size 
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple)), \
			"argument output_size is not of type int or tuple" 
		self.output_size = output_size

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		if isinstance(self.output_size, int):
			new_h, new_w = self.output_size, self.output_size
		else:
			new_h, new_w = self.output_size

		image = cv2.resize(image, (new_w, new_h))

		return {'image': image, 'label': label}


class RandomCrop(object):
	""" Crop randomly the image in a sample 
	
	Args: 
		output_size (tuple or int): Desired output size
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple)), \
			"argument output_size is not of type int or tuple"
		self.output_size = output_size

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		height, width = image.shape[:2]
		if isinstance(self.output_size, int):
			new_height, new_width = self.output_size, self.output_size
		else:
			new_height, new_width = self.output_size

		top = np.random.randint(0, height - new_height)
		left = np.random.randint(0, width - new_width)

		image = image[top:top + new_height, left:left + new_width]

		return {'image': image, 'label': label}
