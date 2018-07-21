import json
from data_loader import CifarDataLoader, CifarDataset, ToTensor

import torch
from torch.utils.data import DataLoader 
from torchvision import transforms

def main():
	with open("config.json", "r") as f:
		config = json.load(f)

	data = CifarDataLoader(config)

	train_data_transform = transforms.Compose([ToTensor()])
	train_data_transformed = CifarDataset(config, data.X_train, data.y_train, 
		transform=train_data_transform)

	# print("data.X_train[9].shape: ", data.X_train[9].shape)

	for i in range(4): 
		sample = train_data_transformed[i]
		print(i, sample['image'].size(), sample['label'].size())


if __name__ == '__main__':
	main()