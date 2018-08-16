import json
from data_loader import CifarDataset, CifarDataLoader
from transformations import ToTensor, ToGrayscale, Normalize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import LeNet

import logging
from trainer import Trainer

def main():

	with open("config.json", "r") as f:
		config = json.load(f)

	## Prepare data
	data = CifarDataset(config)

	all_transforms = transforms.Compose([
		ToGrayscale(), 
		Normalize(), 
		ToTensor()])

	train_data_transformed = CifarDataLoader(config, data.X_train, data.y_train, 
		transform=all_transforms)
	train_loader = DataLoader(train_data_transformed, 
		batch_size=config["data_loader"]["batch_size"], 
		shuffle=False, 
		num_workers=4)

	if config["validation"]["split"]:
		valid_data_transformed = CifarDataLoader(config, data.X_valid, data.y_valid, 
			transform=all_transforms)
		valid_loader = DataLoader(valid_data_transformed, 
			batch_size=config["data_loader"]["batch_size"], 
			shuffle=False, 
			num_workers=4)		
	
	test_data_transformed = CifarDataLoader(config, data.X_test, data.y_test, 
		transform=all_transforms)
	test_loader = DataLoader(test_data_transformed, 
		batch_size=config["data_loader"]["batch_size"], 
		shuffle=False, 
		num_workers=4)

	## Create neural net
	net = LeNet()

	## Training 
	trainer = Trainer(model=net,
		config=config, 
		resume=None, 
		train_data_loader=train_loader, 
		valid_data_loader=valid_loader
		)
	trainer.train()

	# def __init__(self, model, loss, resume, config, 
	# 	data_loader, valid_data_loader=None, train_logger=None):


if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

	main()