import os
import csv
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image

from helpers import csv_save_to_file, csv_format
from model import CNN

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)  # read cvs file into variable
        self.img_dir = img_dir  # folder containing images
        self.transform = transform  
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])  # [row, col]
        image = read_image(img_path)
        """
        csv file is formatted as 3 columns
        [image label], [linear velocity], [angular velocity]
        label = [linear] [angular]
        """
        x = self.img_labels.iloc[idx, 1]  # column 1 --> linear velocity
        z = self.img_labels.iloc[idx, 2]  # column 2 --> angular velocity
        label = csv_format(x, z)
        # print(f"linear velocity: {label[0]}, angular velocity: {label[1]}")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__ == "__main__":
    # setup csv file
    csv_save_to_file()

    """
    Save labels.csv file from computer into varaible
    Save image folder from computer into variable
    Use image path and labels to to create customized dataloader

    """
    annotations_file = "/home/aj/images/labels/avoid_walls_labels.csv"
    img_dir = '/home/aj/images/avoid_walls/'
    training_data = CustomImageDataset(annotations_file, img_dir)
    # training_data.__getitem__(0)
    # exit()
    
    # print(training_data.__len__()) 
    # print(training_data.__getitem__(0))

    # LOAD MODEL
    learning_rate = 0.01
    weight_decay = 0.01
    model = CNN(image_channels=3, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    """
    NOTES:
    - Mean squared error is used to calculate the difference between prediction and label
    - Number of training epochs describes the number of times to train over entire dataset
    """

    mse = nn.MSELoss()  # mean squared error loss class
    num_training_epochs = 100 # loop through training data 100 times
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)  # load images/labels into 

    # TRAIN MODEL
    for epoch in range(num_training_epochs):  # loop through data 100 times
        data_iter = iter(train_dataloader)
        for train_features, train_labels in data_iter:
            # print(len(train_features))  # images --> batch size = 64
            # print(len(train_labels))    # labels --> batch size = 64
            # prediction is a tensor with N,2
            print(train_labels[0])
            prediction = model(train_features)
            # loss is a scalar
            loss = mse(prediction, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())

