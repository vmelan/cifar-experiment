import json
from data_loader import CifarDataset, CifarDataLoader
from transformations import ToTensor, ToGrayscale, Normalize
import torch
from torch.utils import data
from torchvision import transforms

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

	# Sanity check
	for i in range(4): 
		sample = train_data_transformed[i]
		print(i, sample['image'].size(), sample['label'].size())

	train_loader = data.DataLoader(train_data_transformed, 
		batch_size=config["batch_size"], 
		shuffle=False, 
		num_workers=4)
	valid_loader = data.DataLoader(valid_data_transformed, 
		batch_size=config["batch_size"], 
		shuffle=False, 
		num_workers=4)
	test_loader = data.DataLoader(test_data_transformed, 
		batch_size=config["batch_size"], 
		shuffle=False, 
		num_workers=4)


if __name__ == '__main__':
	main()