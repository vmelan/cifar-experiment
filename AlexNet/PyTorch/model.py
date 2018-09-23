import torch 
import torch.nn as nn 
import torch.nn.functional as F


class AlexNet(nn.Module):

	def __init__(self):
		super(AlexNet, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels=3, 
				out_channels=96, 
				kernel_size=(4, 4), 
				stride=(4, 4), 
				padding=0), 
			nn.ReLU()
			)
		self.lrn1 = nn.LocalResponseNorm(size=2, alpha=2e-5, beta=0.75, k=1)
		self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
		self.conv2 = nn.Sequential(
			nn.Conv2d(
				in_channels=96, 
				out_channels=256, 
				kernel_size=(5, 5), 
				stride=(1, 1), 
				padding=1), 
			nn.ReLU()
			)
		self.lrn2 = nn.LocalResponseNorm(size=2, alpha=2e-5, beta=0.75, k=1)
		self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
		self.conv3 = nn.Sequential(
			nn.Conv2d(
				in_channels=256, 
				out_channels=384, 
				kernel_size=(3, 3),
				stride=(1, 1), 
				padding=1), 
			nn.ReLU()
			)
		self.conv4 = nn.Sequential(
			nn.Conv2d(
				in_channels=384,
				out_channels=384, 
				kernel_size=(3, 3),
				stride=(1, 1),
				padding=1),
			nn.ReLU()
			)
		self.conv5 = nn.Sequential(
			nn.Conv2d(
				in_channels=384, 
				out_channels=256, 
				kernel_size=(3, 3), 
				stride=(1, 1), 
				padding=1), 
			nn.ReLU()
			)
		self.dense1 = nn.Sequential(
			nn.Linear(in_features=9216, out_features=4096),
			nn.ReLU()
			)
		self.dense2 = nn.Sequential(
			nn.Linear(in_features=4096, out_features=4096),
			nn.ReLU()
			)
		self.dense3 = nn.Linear(in_features=4096, out_features=10)



	def forward(self, x):
		""" Forward pass of network """
		x = self.conv1(x)
		x = self.lrn1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.lrn2(x)
		x = self.pool2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = x.view(x.size(0), -1) # flatten to size (batch_size, 6 x 6 x 256)
		x = self.dense1(x)
		x = self.dense2(x)
		x = self.dense3(x)

		return x 

if __name__ == '__main__':
	net = AlexNet()
	print(net)