import os
import logging
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

class Trainer():

	def __init__(self, model, config, resume, 
		train_data_loader, valid_data_loader=None, test_data_loader=None):

		self.config = config
		self.model = model

		self.train_data_loader = train_data_loader 
		self.valid_data_loader = valid_data_loader 
		self.valid = True if self.valid_data_loader else False

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
		self.optimizer = getattr(optim, config['optimizer_type'])(model.parameters(), **config["optimizer"])
		# Get criterion for computing loss
		self.criterion = nn.MSELoss()

		self.start_epoch = 0
		self.checkpoint_dir = os.path.join(config["trainer"]["save_dir"], config["experiment_name"])

		# Create writer for tensorboard visualization
		self.writer = SummaryWriter('tensorboard/' + config["experiment_name"] + "/") # path to log files

		# Start from a checkpoint
		if resume: 
			self._resume_checkpoint(resume)


	def train(self):
		""" Training procedure """

		for epoch in range(self.start_epoch, self.config["trainer"]["epochs"]):
			train_loss, train_acc = self._train_epoch(epoch)
			if self.valid:
				val_loss, val_acc = self._valid_epoch(epoch)

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
			loss = self.criterion(pred_labels, true_labels)
			# backward pass to calculate the weight gradients
			loss.backward()
			# update the weights
			self.optimizer.step()

			# track loss and accuracy
			train_loss += loss.item()
			train_acc += self._compute_accuracy(true_labels, pred_labels)

			# add graph to writer 
			if (epoch == 0 and batch_idx == 0): self.writer.add_graph(self.model, (images,))


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
				val_loss += self.criterion(pred_labels, true_labels).item()
				val_acc += self._compute_accuracy(true_labels, pred_labels)

		val_loss = val_loss / len(self.valid_data_loader)
		val_acc = val_acc / len(self.valid_data_loader)

		return val_loss, val_acc


	def _to_tensor(self, sample):
		""" 
		Convert batch of images and labels to FloatTensor and
		move them to GPU if CUDA is available
		"""
		images, true_labels = sample['image'].type(torch.FloatTensor), sample['label'].type(torch.FloatTensor)
		if self.with_cuda:
			images, true_labels = images.to(self.gpu), true_labels.to(self.gpu)
		return images, true_labels


	def _compute_accuracy(self, true_labels, pred_labels):
		""" Compute accuracy between true and predicted labels """
		accuracy = torch.sum(torch.argmax(true_labels, dim=1) == torch.argmax(pred_labels, dim=1))
		# mean across the batches
		accuracy = accuracy.numpy() / self.config["data_loader"]["batch_size"] 
		return accuracy

	def evaluate(self):
		""" Evaluate of test data """

		# Evaluation mode
		self.model.eval()

		test_loss, test_acc = 0.0, 0.0

		with torch.no_grad():
			for batch_idx, sample in enumerate(self.test_data_loader):
				images, true_labels = self._to_tensor(sample)
				pred_labels = self.model.forward(images)
				test_loss += self.criterion(pred_labels, true_labels).item()
				test_acc += self._compute_accuracy(true_labels, pred_labels)

		test_loss = test_loss / len(self.test_data_loader)
		test_acc = test_acc / len(self.test_data_loader)

		self.logger.info("test_loss= {:.3f}, test_acc= {:.3f}".format(test_loss, test_acc))
