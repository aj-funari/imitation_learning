import os
import csv
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
import glob
from helpers import image_label_to_tensor

class CustomImageDataset(Dataset):
    # def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    def __init__ (self, transform=None, target_transform=None):
        # self.img_labels = pd.read_csv(annotations_file)  # read cvs file
        # self.img_dir = img_dir  # folder containing images
        # self.dir = '/home/aj/catkin_ws/src/imitation_learning/avoid_walls'/
        # self.dir = '/home/aj/catkin_ws/src/imitation_learning/collision_walls'
        self.dir = '/home/aj/catkin_ws/src/imitation_learning/images'
        self.transform = transform  
        self.target_transform = target_transform
        
        # junhong modification
        # self.all_image_names = glob.glob("path/to/your/image/folder")
        # self.all_image_names = glob.glob('/home/aj/catkin_ws/src/imitation_learning/images')

        # aj modification
        self.all_image_names = os.listdir(self.dir) 
        
    def __len__(self):
        # junhong modification
        return len(self.all_image_names)

        # original function
        # return len(self.img_labels)

    def __getitem__(self, idx):
        # junhong modification
        # label = self.your_parsing_function(self.all_image_names[idx])
        # image = read_image(self.all_image_names[idx])
        # return image, label

        # aj modification
        label = self.all_image_names[idx]  # '1.4502599239349365-1.095364272594452-19:08:20-.jpeg'
        img_path = os.path.join(self.dir, label)  # path to image
        image = read_image(img_path)
        image = image / 255  # normalize image --> change values from [0,255] to [0,1]

        label = image_label_to_tensor(label)  # return type tensor --> tensor([[x, z]))

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

        """
        function .iloc --> helps us select a specific cell of csv file
        - In this situation, we are requesting the image labels. They 
          located in the first column of the csv file
        """

        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])  # [row, col]
        # image = read_image(img_path)  # converts image to tensor
        # # print(self.img_labels.iloc[idx, 0])
        
        # image = image / 255.0  # normalize image: divide by 255.0  

        # """
        # csv file is formatted as 3 columns
        # [image label], [linear velocity], [angular velocity]
        # label = [linear] [angular]
        # """

        # x = self.img_labels.iloc[idx, 1]  # col 1: linear velocity --> type string
        # z = self.img_labels.iloc[idx, 2]  # col 2: angular velocity --> type string
        # label = csv_label_to_tensor(x, z)  # e.g. tensor([2.0000, 0.8548])

        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        # return image, label  # returns tensor image and cooresponding label in a tuple
