import os
import csv
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image

from helpers import csv_label_to_tensor


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)  # read cvs file
        self.img_dir = img_dir  # folder containing images
        self.transform = transform  
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):      # function called every time image needs to be referenced
        """
        function .iloc --> helps us select a specific cell of csv file
        - In this situation, we are requesting the image labels. They 
          located in the first column of the csv file
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])  # [row, col]
        image = read_image(img_path)  # converts image to tensor
        # print(self.img_labels.iloc[idx, 0])
        
        image = image / 255.0  # normalize image: divide by 255.0  

        """
        csv file is formatted as 3 columns
        [image label], [linear velocity], [angular velocity]
        label = [linear] [angular]
        """

        x = self.img_labels.iloc[idx, 1]  # col 1: linear velocity --> type string
        z = self.img_labels.iloc[idx, 2]  # col 2: angular velocity --> type string
        label = csv_label_to_tensor(x, z)  # e.g. tensor([2.0000, 0.8548])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label  # returns tensor image and cooresponding label in a tuple