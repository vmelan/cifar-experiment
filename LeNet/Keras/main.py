import warnings
warnings.filterwarnings("ignore")

import json
from data_loader import DataLoader
import model
# from model import gaussian
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import model_from_json

def main():
	with open("config.json", "r") as f:
		config = json.load(f)

	# Cifar data 
	data = DataLoader(config)

	# Create object LeNet
	LeNet = model.LeNet(config)

	# Building LeNet model
	LeNet_model = LeNet.build_model()

	# Compile Model 
	LeNet_model.compile(optimizer=Adam(lr=config["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=1e-7), 
		loss='categorical_crossentropy', 
		metrics=['accuracy'])

	## Create callbacks
	# checkpoint for validation data improvement
	checkpoint_callback = ModelCheckpoint(config["model_checkpoint_path"], \
		monitor="val_acc", 
		verbose=0, 
		save_best_only=True,
		mode="max")
	# Tensorboard
	tensorboard_callback = TensorBoard(log_dir="./logs", 
		histogram_freq=1, 
		batch_size=config["batch_size"], 
		write_graph=True)

	## Train the model
	# LeNet.fit(data.X_train, data.y_train, 
	# 	validation_data=(data.X_valid, data.y_valid),
	# 	epochs=config["num_epochs"], 
	# 	batch_size=config["batch_size"],
	# 	callbacks=[checkpoint_callback, tensorboard_callback], 
	# 	verbose=1
	# 	)

	LeNet_model.fit_generator(generator=(data.next_batch()),
		validation_data=(data.X_valid, data.y_valid),  
		epochs=config["num_epochs"],
		steps_per_epoch=config["num_iter_per_epoch"], 
		callbacks=[checkpoint_callback, tensorboard_callback],
		verbose=1
		)

	## Saving model and weights
	with open("./saved/model.json", "w") as json_file:
		json_file.write(LeNet_model.to_json())
	LeNet_model.save_weights("saved/model_weights.hdf5")
	print("LeNet model and weights saved")

	## Load model and weights for validation
	with open("./saved/model.json", "r") as json_file:
		loaded_model_json = json_file.read()
	LeNet = model_from_json(loaded_model_json)

	# Compile Model needed after model loaded
	LeNet.compile(optimizer=Adam(lr=config["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=1e-7), 
		loss='categorical_crossentropy', 
		metrics=['accuracy'])

	## Evaluate on test data
	results = LeNet_model.evaluate(x=data.X_test, y=data.y_test, 
		batch_size=config["batch_size"], verbose=1)


	print(results)

if __name__ == '__main__':
	main()