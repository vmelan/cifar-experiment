import os
import logging
from keras.callbacks import ModelCheckpoint, TensorBoard

from tqdm import tqdm

class Trainer(object):

	def __init__(self, model, data, config):
		self.model = model
		self.data = data 
		self.config = config
		self.callbacks = []

		# Validation data
		if config["validation"]["split"]:
			self.validation_data = (self.data.X_valid, self.data.y_valid)
		else: 
			self.validation_data = None

		# Logging for this class
		self.logger = logging.getLogger(self.__class__.__name__)

		# Populate the callbacks into the self.callbacks list
		self.init_callbacks()

	def init_callbacks(self):
		"""
		Create callbacks 
		"""

		save_ckpt_path = self.config["trainer"]["save_dir"] + self.config["experiment_name"] + "/" 
		if not os.path.exists(save_ckpt_path):
			os.makedirs(save_ckpt_path)
		save_name = self.config["trainer"]["save_trained_name"] + "_{epoch:03d}.hdf5"
		self.callbacks.append(
			ModelCheckpoint(save_ckpt_path + save_name, 
				save_best_only=False,
				mode="auto",
				period=self.config["trainer"]["save_freq"], 
				verbose=0,
				)
			)

		self.callbacks.append(
			TensorBoard(
				log_dir="tensorboard/" + self.config["experiment_name"], 
				histogram_freq=1, 
				batch_size=self.config["trainer"]["batch_size"],
				write_graph=True, 
				)
			)


	def train(self):
		"""
		Train the model
		"""
		if self.config["trainer"]["verbose"]:
			self.model.fit_generator(generator=(self.data.next_batch()),
				validation_data=(self.data.X_valid, self.data.y_valid),  
				epochs=self.config["trainer"]["epochs"],
				steps_per_epoch=self.config["trainer"]["num_iter_per_epoch"], 
				callbacks=self.callbacks,
				verbose=1
				)
		else:
			for epoch in tqdm(range(self.config["trainer"]["epochs"])):
				self.model.fit_generator(generator=(self.data.next_batch()),
					validation_data=(self.data.X_valid, self.data.y_valid),  
					epochs=1,
					steps_per_epoch=self.config["trainer"]["num_iter_per_epoch"], 
					callbacks=self.callbacks,
					verbose=0
					)			

		self.logger.info("Training complete")


	def evaluate(self):
		"""
		Evaluate test set
		"""
		results = self.model.evaluate(
			x=self.data.X_test, 
			y=self.data.y_test, 
			batch_size=self.config["trainer"]["batch_size"], 
			verbose=0
			)

		self.logger.info("test loss = {:.4f}".format(results[0]))
		self.logger.info("test accuracy = {:.4f}".format(results[1]))


	def save_weights(self):
		""" Save network weights """
		if self.model is None:
			self.logger.warning("You have to build the model first")
		# Serialize weights to hdf5
		save_path = self.config["trainer"]["save_dir"] + self.config["experiment_name"] + \
			"/" + self.config["trainer"]["save_trained_name"]
		self.model.save_weights(save_path + "_full.hdf5")

		self.logger.info("Model weights saved")


	def load_weights(self, checkpoint_path):
		""" Load model weights """
		if self.model is None:
			self.logger.warning("You have to build the model first")		

		self.logger.info("Loading model checkpoint {:s} ... \n".format(checkpoint_path))
		self.model.load_weights(checkpoint_path)
		self.logger.info("Model weights loaded")