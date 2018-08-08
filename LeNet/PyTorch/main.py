import json
from tqdm import tqdm
from data_loader import CifarDataset, CifarDataLoader
from transformations import ToTensor, ToGrayscale, Normalize
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import LeNet


def main():
	with open("config.json", "r") as f:
		config = json.load(f)

	data = CifarDataset(config)

	all_transforms = transforms.Compose([
		ToGrayscale(), 
		Normalize(), 
		ToTensor()])

	train_data_transformed = CifarDataLoader(config, data.X_train, data.y_train, 
		transform=all_transforms)
	valid_data_transformed = CifarDataLoader(config, data.X_valid, data.y_valid, 
		transform=all_transforms)
	test_data_transformed = CifarDataLoader(config, data.X_test, data.y_test, 
		transform=all_transforms)

	train_loader = DataLoader(train_data_transformed, 
		batch_size=config["batch_size"], 
		shuffle=False, 
		num_workers=4)
	valid_loader = DataLoader(valid_data_transformed, 
		batch_size=config["batch_size"], 
		shuffle=False, 
		num_workers=4)
	test_loader = DataLoader(test_data_transformed, 
		batch_size=config["batch_size"], 
		shuffle=False, 
		num_workers=4)

	## Define neural network
	net = LeNet()

	## Define the loss and optimization
	criterion = nn.MSELoss()
	optimizer = optim.Adam(net.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999), eps=1e-08)

	## Create writer for tensorboard visualization
	writer = SummaryWriter('tensorboard/' + config["experiment_name"] + "/") # path to log files
	
	## Training net
	for epoch in range(config["num_epochs"]):
		# training mode
		net.train()

		total_loss = 0.0
		total_accuracy = 0.0

		for batch_idx, sample in tqdm(enumerate(train_loader), desc="epoch [" + str(epoch + 1) + "/" + str(config["num_epochs"]) + "]"):
			images, true_labels = sample['image'].type(torch.FloatTensor), sample['label'].type(torch.FloatTensor)
			# zero the parameter (weight) gradients
			optimizer.zero_grad()
			# compute forward pass of images through the net 
			pred_labels = net.forward(images)
			# compute the loss between predicted and true labels
			loss = criterion(pred_labels, true_labels)
			# backward pass to calculate the weight gradients
			loss.backward()
			# update the weights
			optimizer.step()
			# track loss for print
			total_loss += loss.item() # loss.item() gets the a scalar value held in the loss
			# compute accuracy for print  
			total_accuracy += torch.sum(torch.argmax(true_labels, dim=1) == torch.argmax(pred_labels, dim=1))

		if (epoch % config["display_step"] == 0):
			# evaluation mode
			net.eval()

			val_acc = 0.0
			val_loss = 0.0

			for batch_idx, sample in enumerate(valid_loader):
				images, true_labels = sample['image'].type(torch.FloatTensor), sample['label'].type(torch.FloatTensor)
				pred_labels = net.forward(images)
				val_loss += criterion(pred_labels, true_labels).item()
				val_acc += torch.sum(torch.argmax(true_labels, dim=1) == torch.argmax(pred_labels, dim=1))

			# Prepare variables for print and tensorboard
			train_loss = total_loss / len(train_loader)
			train_acc = total_accuracy.numpy() / len(data.X_train)
			val_loss = val_loss / len(valid_loader)
			val_acc = val_acc.numpy() / len(data.X_valid)

			if config["verbose"]: 
				print("Epoch: %03d," % (epoch + 1), 
					"train_loss= %.3f," % (train_loss), 
					"train_acc= %.3f," % (train_acc), 
					"val_loss= %.3f," % (val_loss),
					"val_acc=%.3f" % (val_acc))

			# Write scalars
			writer.add_scalar('train_loss', train_loss, epoch)
			writer.add_scalar('train_acc', train_acc, epoch)
			writer.add_scalar('val_loss', val_loss, epoch)
			writer.add_scalar('val_acc', val_acc, epoch)

	print("Training complete")

	## Saving model parameters
	torch.save(net.state_dict(), config["model_checkpoint_path"])
	print("Model parameters saved !")

	## Loading model and model parameters
	net = LeNet()
	net.load_state_dict(torch.load(config["model_checkpoint_path"]))

	## Inference 
	net.eval()
	test_loss, test_acc = 0.0, 0.0
	for batch_idx, sample in tqdm(enumerate(test_loader), desc="Inference"):
		images, true_labels = sample['image'].type(torch.FloatTensor), sample['label'].type(torch.FloatTensor)
		pred_labels = net.forward(images)
		test_loss += criterion(pred_labels, true_labels).item()
		test_acc += torch.sum(torch.argmax(true_labels, dim=1) == torch.argmax(pred_labels, dim=1))
	test_loss = test_loss / len(test_loader)
	test_acc = test_acc.numpy() / len(data.X_test)
	print("test_loss: %.3f" % (test_loss), "test_acc: %.3f" % (test_acc))


if __name__ == '__main__':
	main()