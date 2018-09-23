import json 
import logging 
from torch.utils.data import DataLoader 
from torchvision import transforms 
from model import AlexNet 
from trainer import Trainer 
from data_loader import CifarDataset, CifarDataLoader
from transformations import * 

from matplotlib import pyplot as plt

import pdb

def main():
	## Open config file 
	with open("config.json", "r") as f:
		config = json.load(f)

	## Prepare data
	data = CifarDataset(config)

	train_transforms = transforms.Compose([
		Normalize(), 
		Resize((256, 256)), 
		RandomCrop((227, 227)), 
		ToTensor()])

	eval_transforms = transforms.Compose([
		Normalize(), 
		Resize((227, 227)), 
		ToTensor()])

	train_data_transformed = CifarDataLoader(config, data.X_train, data.y_train,
		transform=train_transforms)
	train_loader = DataLoader(train_data_transformed, 
		batch_size=config["data_loader"]["batch_size"],
		shuffle=True, 
		num_workers=0)

	if config["validation"]["split"]:
		valid_data_transformed = CifarDataLoader(config, data.X_valid, data.y_valid, 
			transformed=eval_transforms)
		valid_loader = DataLoader(valid_data_transformed, 
			batch_size=config["data_loader"]["batch_size"], 
			shuffle=False, 
			num_workers=0)
	else:
		valid_loader = None

	test_data_transformed = CifarDataLoader(config, data.X_test, data.y_test, 
		transform=eval_transforms)
	test_loader = DataLoader(test_data_transformed, 
		batch_size=config["data_loader"]["batch_size"], 
		shuffle=False, 
		num_workers=0)

	## Create neural net 
	net = AlexNet()

	## Training 
	trainer = Trainer(model=net, 
		config=config, 
		train_data_loader=train_loader,
		valid_data_loader=valid_loader, 
		test_data_loader=test_loader
		)
	trainer.train()

	## Saving model parameters
	trainer.save_model_params()

	## Evaluate test data
	trainer.evaluate()


if __name__ == '__main__':
	# set logging config 
	logging.basicConfig(level=logging.DEBUG, format="line %(line)d: %(message)s")

	main()