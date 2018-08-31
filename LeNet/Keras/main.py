import json
import logging
from data_loader import DataLoader
from model import LeNet
from trainer import Trainer 

def main():
	with open("config.json", "r") as f:
		config = json.load(f)

	# Load Cifar data 
	data = DataLoader(config)

	# Create LeNet model
	net = LeNet(config)

	# Create trainer
	trainer = Trainer(net.model, data, config)

	# # Train model
	# trainer.train()

	# # Save LeNet model weights
	# trainer.save_weights()

	# Load weights
	load_path = self.config["trainer"]["save_dir"] + self.config["experiment_name"] + \
			"/" + self.config["trainer"]["save_trained_name"] +  "_full.hdf5"
	trainer.load_weights(load_path)

	# Evaluate validation set
	trainer.evaluate()

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')

	main()