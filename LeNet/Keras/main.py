import json
from data_loader import DataLoader
import model
from keras.optimizers import Adam


def main():
	with open("config.json", "r") as f:
		config = json.load(f)

	# Cifar data 
	data = DataLoader(config)

	# Create model
	LeNet = model.LeNet(config)

	# Building LeNet model
	LeNet.build_model()

	# Compile Model 
	LeNet.compile(optimizer=Adam(lr=config["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=1e-7), 
		loss='categorical_crossentropy', 
		metrics=['accuracy'])

	# Create callbacks