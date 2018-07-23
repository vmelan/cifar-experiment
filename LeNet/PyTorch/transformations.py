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


class ToGrayscale(object):
	""" Conver a color image to grayscale """

	def __call__(self, sample):
		image, label = sample['image'], sample['label']

		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray_image = gray_image.reshape(32, 32, 1)

		return {'image': gray_image, 'label': label}