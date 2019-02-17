import json
import logging
from data_loader import DataLoader
from model import AlexNet
from trainer import Trainer

import pdb
import os
## GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
	with open("config.json", "r") as f:
		config = json.load(f)

	# Load Cifar
	data = DataLoader(config)

	# Create AlexNet model
	net = AlexNet(config)

	# pdb.set_trace()

	# Create trainer
	trainer = Trainer(net.model, data, config)

	# Train model
	trainer.train()

	# Save model weights
	trainer.save_model()

	# Load weights
	net.model = trainer.load_model(config["load_model"])

	# Evaluation test set
	trainer.evaluate()

if __name__ =='__main__':
	logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')

	main()
