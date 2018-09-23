import os
import logging
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import pdb

class Trainer():

	def __init__(self, model, config, 
		train_data_loader, valid_data_loader=None, test_data_loader=None):

		self.config = config
		self.model = model

		self.train_data_loader = train_data_loader 
		self.valid_data_loader = valid_data_loader 
		self.valid = True if self.valid_data_loader else False
		self.test_data_loader = test_data_loader

		# Logger for this class
		self.logger = logging.getLogger(self.__class__.__name__)

		self.with_cuda = config["cuda"] and torch.cuda.is_available()
		# move network to gpu if possible
		if config["cuda"] and not torch.cuda.is_available():
			self.logger.warning("Warning: There's no CUDA support on this machine,"
				"training is performed on CPU.")
		elif not config["cuda"] and torch.cuda.is_available():
			self.logger.info("Training is performed on CPU by user's choice")
		else:
			self.device = torch.device('cuda:' + str(config['gpu']))
			self.model = self.model.to(self.device)

		# Get optimizer
		self.optimizer = getattr(optim, config["optimizer"]['optimizer_type'])(model.parameters(), 
			**config["optimizer"]["optimizer_params"])
		
		# Learning rate scheduler
		if (config["scheduler"]["use_scheduler"]):
			self.lr_scheduler = getattr(optim.lr_scheduler, config["scheduler"]["lr_scheduler_type"])
			self.lr_scheduler = self.lr_scheduler(self.optimizer, **config["scheduler"]["scheduler_params"])
			self.lr_scheduler_freq = config["scheduler"]["lr_scheduler_freq"]
		else:
			self.lr_scheduler = None
			self.lr_scheduler_freq = None
		
		# Get criterion for computing loss
		self.criterion = nn.CrossEntropyLoss()

		self.start_epoch = 0

		# Create writer for tensorboard visualization
		self.writer = SummaryWriter('tensorboard/' + config["experiment_name"] + "/") # path to log files


	def train(self):
		""" Training procedure """

		for epoch in range(self.start_epoch, self.config["trainer"]["epochs"]):
			# train and evaluate 
			train_loss, train_acc = self._train_epoch(epoch)
			if self.valid:
				val_loss, val_acc = self._valid_epoch()

			# use scheduler 
			if (self.lr_scheduler and ((epoch + 1) % self.lr_scheduler_freq == 0)):
				self.lr_scheduler.step(epoch)
				lr = self.lr_scheduler.get_lr()[0]
				self.logger.info("New learning Rate: {:.6f}".format(lr))

			# print performance to logger
			if self.config["trainer"]["verbose"]:
				if self.valid:
					self.logger.info("Epoch: {:03d}, "
						"train_loss= {:.3f}, train_acc= {:.3f}, "
						"val_loss= {:.3f}, val_acc= {:.3f}".format(
							epoch+1, train_loss, train_acc, val_loss, val_acc) 
						)
				else:
					self.logger.info("Epoch: {:03d}, "
						"train_loss= {:.3f}, train_acc= {:.3f}".format(
							epoch+1, train_loss, train_acc) 
						)			

			# Add performance to writer 
			self.writer.add_scalar('train_loss', train_loss, epoch)
			self.writer.add_scalar('train_acc', train_acc, epoch)
			if self.valid:
				self.writer.add_scalar('val_loss', val_loss, epoch)
				self.writer.add_scalar('val_acc', val_acc, epoch)

		self.logger.info("Training complete")


	def _train_epoch(self, epoch):
		""" Training for an epoch """

		# Training mode 
		self.model.train()

		train_loss, train_acc = 0.0, 0.0

		for batch_idx, sample in tqdm(enumerate(self.train_data_loader)):
			images, true_labels = self._to_tensor(sample)
			# zero the parameter (weight) gradients
			self.optimizer.zero_grad()
			# compute forward pass of images through the net
			pred_labels = self.model.forward(images)
			# compute the loss between predicted and true labels
			loss = self.criterion(pred_labels, torch.argmax(true_labels, 1))
			# backward pass to calculate the weight gradients
			loss.backward()
			# update the weights
			self.optimizer.step()

			# track loss and accuracy
			train_loss += loss.item()
			train_acc += self._compute_accuracy(true_labels, pred_labels)

			# add graph to writer 
			if (epoch == self.start_epoch and batch_idx == 0): self.writer.add_graph(self.model, (images,))

		train_loss = train_loss / len(self.train_data_loader)
		train_acc = train_acc / len(self.train_data_loader)

		return train_loss, train_acc


	def _valid_epoch(self):
		""" Evaluation of validation data for an epoch """

		# Evaluation mode
		self.model.eval()

		val_loss, val_acc = 0.0, 0.0

		with torch.no_grad():
			for batch_idx, sample in enumerate(self.valid_data_loader):
				images, true_labels = self._to_tensor(sample)
				pred_labels = self.model.forward(images)
				val_loss += self.criterion(pred_labels, torch.argmax(true_labels, 1)).item()
				val_acc += self._compute_accuracy(true_labels, pred_labels)

		val_loss = val_loss / len(self.valid_data_loader)
		val_acc = val_acc / len(self.valid_data_loader)

		return val_loss, val_acc


	def _to_tensor(self, sample):
		""" 
		Convert batch of images to FloatTensor and labels to LongTensor
		and move them to GPU if CUDA is available
		"""
		images = sample['image'].type(torch.FloatTensor)
		true_labels = sample['label'].type(torch.LongTensor)
		if self.with_cuda:
			images, true_labels = images.to(self.device), true_labels.to(self.device)
		# images, true_labels = images, true_labels
		return images, true_labels


	def _compute_accuracy(self, true_labels, pred_labels):
		""" Compute accuracy between true and predicted labels """
		accuracy = torch.sum(torch.argmax(true_labels, dim=1) == torch.argmax(pred_labels, dim=1))
		# mean across the batches
		accuracy = accuracy.numpy() / self.config["data_loader"]["batch_size"] 
		return accuracy


	def evaluate(self):
		""" Evaluation of test data """

		# Evaluation mode
		self.model.eval()

		test_loss, test_acc = 0.0, 0.0

		with torch.no_grad():
			for batch_idx, sample in tqdm(enumerate(self.test_data_loader), desc="Inference"):
				images, true_labels = self._to_tensor(sample)
				pred_labels = self.model.forward(images)
				test_loss += self.criterion(pred_labels, torch.argmax(true_labels, 1)).item()
				test_acc += self._compute_accuracy(true_labels, pred_labels)

		test_loss = test_loss / len(self.test_data_loader)
		test_acc = test_acc / len(self.test_data_loader)

		self.logger.info("test_loss= {:.3f}, test_acc= {:.3f}".format(test_loss, test_acc))


	def save_model_params(self):
		""" Save model parameters after training """

		save_path = os.path.join(self.config["trainer"]["save_dir"], 
			self.config["experiment_name"])
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		torch.save(self.model.state_dict(), save_path + "/" + self.config["trainer"]["save_trained_name"])
		self.logger.info("Model parameters saved")