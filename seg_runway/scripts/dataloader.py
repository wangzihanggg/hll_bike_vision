import os
import numpy as np
import cv2

from utils.transform import train_transform
from torch.utils.data import Dataset
from PIL import Image
from utils.resize_image_keep_scale import resize_image_fixed

img_label_size = (320, 240)
# label = Image.open(self.image_paths[index].replace('images', 'labels').replace('jpg', 'png')).convert('RGB')
# label = label.convert("L")  # delete if you want train in three channel image input

class UnetDataset(Dataset):
	def __init__(self, data_path, train_transform=None):
		super(UnetDataset, self).__init__()
		self.data_path = os.path.join(data_path, "images")
		self.image_names = os.listdir(self.data_path)
		self.train_transform = train_transform
		self.image_paths = [os.path.join(self.data_path, image_name) for image_name in self.image_names]

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, index):
		img_path = self.image_paths[index]
		label_path = self.image_paths[index].replace('images', 'labels').replace('jpg', 'png')
		if not os.path.exists(label_path):
			temp_label = np.zeros((640, 480, 1), dtype=np.uint8)
		else:
			temp_label = cv2.imread(label_path)
		temp_label = cv2.resize(temp_label, img_label_size)
		if len(temp_label.shape) == 3:
			gray = cv2.cvtColor(temp_label, cv2.COLOR_BGR2GRAY)
		else:
			gray = temp_label
		ret_val, label = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

		image = Image.open(img_path).convert('RGB')

		image = resize_image_fixed(image, size=img_label_size)

		image = self.train_transform(image)
		label = self.train_transform(label)
		return image, label


if __name__ == '__main__':
	image_path = 'datasets/train'
	train_dataset = UnetDataset(image_path, train_transform)
	print(train_dataset[0][0])
	print(train_dataset[0][1])
