import os
import logging
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np

class Trainer(object):

	def __init__(self, model, data, config):
		self.model = model
		self.data = data
		self.config = config

		# Validation data
		if config["validation"]["split"]:
			self.validation_data = (self.data.X_valid, self.data.y_valid)
		else:
			self.validation_data = None

		# Logging for this class
		self.logger = logging.getLogger(self.__class__.__name__)

		# Populate the callbacks into the self.callbacks list
		self.callbacks = []
		self.init_callbacks()


	def init_callbacks(self):
		""" Create callbacks """

		# Checkpoint callback
		save_ckpt_path = self.config["trainer"]["save_dir"] + self.config["experiment_name"] + "/"
		if not os.path.exists(save_ckpt_path):
			os.makedirs(save_ckpt_path)
		save_name = self.config["trainer"]["save_trained_name"] + "_{epoch:03d}.hdf5"
		self.callbacks.append(
			ModelCheckpoint(save_ckpt_path + save_name,
				save_best_only=False,
				save_weights_only=False,
				mode="auto",
				period=self.config["trainer"]["save_freq"],
				verbose=0
				)
			)

		# Tensorboard callback
		self.callbacks.append(
			TensorBoard(
				log_dir="tensorboard/" + self.config["experiment_name"],
				histogram_freq=0,
				batch_size=self.config["trainer"]["batch_size"],
				write_graph=True
				)
			)

	def train(self):
		""" Train the model """

		if self.validation_data is not None:
			self.model.fit_generator(
				generator=self.data.next_batch(),
				validation_data=self.validation_data,
				epochs=self.config["trainer"]["epochs"],
				steps_per_epoch=self.config["trainer"]["num_iter_per_epoch"],
				callbacks=self.callbacks,
				verbose=self.config["trainer"]["verbose"]
				)
		else:
			self.model.fit_generator(
				generator=self.data.next_batch(),
				epochs=self.config["trainer"]["epochs"],
				steps_per_epoch=self.config["trainer"]["num_iter_per_epoch"],
				callbacks=self.callbacks,
				verbose=self.config["trainer"]["verbose"]
				)

		self.logger.info("Training complete")


	def evaluate(self):
		""" Evaluate test set """

		results = self.model.evaluate(
			x=self.data.X_test,
			y=self.data.y_test,
			batch_size=self.config["trainer"]["batch_size"],
			verbose=0
			)

		self.logger.info("test loss = {:.4f}".format(results[0]))
		self.logger.info("test accuracy = {:.4f}".format(results[1]))
