import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):

	def __init__(self):
		super(LeNet, self).__init__()

		# Conv layers
		self.conv1 = nn.Conv2d(in_channels=1, 
			out_channels=6, 
			kernel_size=(5, 5), 
			stride=(1, 1), 
			padding=0)

		self.conv2 = nn.Conv2d(in_channels=6, 
			out_channels=16, 
			kernel_size=(5, 5), 
			stride=(1, 1), 
			padding=0)

		self.conv3 = nn.Conv2d(in_channels=16, 
			out_channels=120, 
			kernel_size=(5, 5), 
			stride=(1, 1), 
			padding=0)

		# Average Pooling
		self.avgpool = nn.AvgPool2d(kernel_size=(2, 2), 
			stride=(2, 2), 
			padding=0)

		# Fully connected layers
		self.fc1 = nn.Linear(in_features=1 * 1 * 120, out_features=84)

		self.fc2 = nn.Linear(in_features=84, out_features=10)


	def forward(self, x):
		""" Forward pass of network """

		## Conv layers
		x = self.pool(F.tanh(self.conv1(x)))
		x = self.pool(F.tanh(self.conv2(x)))
		x = self.pool(F.tanh(self.conv3(x)))

		## Flatten
		x = x.view(x.size(0), -1)

		## Fully connected layers
		x = F.tanh(self.fc1(x))
		x = self.fc2(x)

		x = F.softmax(x, dim=1)