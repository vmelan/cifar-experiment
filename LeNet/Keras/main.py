import json
from data_loader import DataLoader
import model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

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
	LeNet.model.compile(optimizer=Adam(lr=config["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=1e-7), 
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
	LeNet.model.fit(data.X_train, data.y_train, 
		validation_data=(data.X_valid, data.y_valid),
		epochs=config["num_epochs"], 
		batch_size=config["batch_size"],
		callbacks=[checkpoint_callback, tensorboard_callback], 
		verbose=1
		)

	# Saving model and weights
	LeNet.model.save_model()
	LeNet.model.save_weights()



if __name__ == '__main__':
	main()