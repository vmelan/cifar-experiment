import json
from data_loader import CifarDataLoader, CifarDataset, ToTensor, ToGrayscale, Normalize

import torch
from torch.utils.data import DataLoader 
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

	# print("data.X_train[9].shape: ", data.X_train[9].shape)

	for i in range(4): 
		sample = train_data_transformed[i]
		print(i, sample['image'].size(), sample['label'].size())

	


if __name__ == '__main__':
	main()