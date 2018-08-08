import json
from data_loader import CifarDataset, CifarDataLoader
from transformations import ToTensor, ToGrayscale, Normalize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import LeNet

def main():
	with open("config.json", "r") as f:
		config = json.load(f)

	data = CifarDataset(config)
	# print("data.X_train.shape: ", data.X_train.shape, "data.y_train.shape", data.y_train.shape)

	all_transforms = transforms.Compose([
		ToGrayscale(), 
		Normalize(), 
		ToTensor()])

	train_data_transformed = CifarDataLoader(config, data.X_train, data.y_train, 
		transform=all_transforms)
	# valid_data_transformed = CifarDataLoader(config, data.X_valid, data.y_valid, 
	# 	transform=all_transforms)
	# test_data_transformed = CifarDataLoader(config, data.X_test, data.y_test, 
	# 	transform=all_transforms)

	# Sanity check
	# for i in range(4): 
	# 	sample = train_data_transformed[i]
	# 	print(i, sample['image'].size(), sample['label'].size())

	train_loader = DataLoader(train_data_transformed, 
		batch_size=config["batch_size"], 
		shuffle=False, 
		num_workers=4)
	# valid_loader = DataLoader(valid_data_transformed, 
	# 	batch_size=config["batch_size"], 
	# 	shuffle=False, 
	# 	num_workers=4)
	# test_loader = DataLoader(test_data_transformed, 
	# 	batch_size=config["batch_size"], 
	# 	shuffle=False, 
	# 	num_workers=4)

	## Define neural network
	net = LeNet()

	## Define the loss and optimization
	criterion = nn.MSELoss()
	optimizer = optim.Adam(net.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999), eps=1e-08)
	
	## Training net
	# training mode
	net.train()
	for epoch in range(config["num_epochs"]):
		total_loss = 0.0
		for batch_idx, data in enumerate(train_loader):
			images, true_labels = data['image'].type(torch.FloatTensor), data['label'].type(torch.FloatTensor)
			# zero the parameter (weight) gradients
			optimizer.zero_grad()
			# compute forward pass of images through the net 
			pred_labels = net.forward(images)
			# compute the loss between predicted and true labels
			loss = criterion(pred_labels, true_labels)
			# backward pass to calculate the weight gradients
			loss.backward()
			# update the weights
			optimizer.step()
			# track loss for print
			total_loss += loss.item() # loss.item() gets the a scalar value held in the loss

		if (epoch % config["display_step"] == 0):
			print("Epoch: %03d, " % (epoch + 1), 
				"loss= %.3f" % (total_loss / len(train_loader)))

	print("Training complete")

if __name__ == '__main__':
	main()