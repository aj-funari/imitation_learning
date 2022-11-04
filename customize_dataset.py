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

    # function called every time image reference needed for training
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])  # [row, col]
        image = read_image(img_path)
        # normalize image: divide by 255.0
        image = image / 255.0  

        """
        csv file is formatted as 3 columns
        [image label], [linear velocity], [angular velocity]
        label = [linear] [angular]
        """

        x = self.img_labels.iloc[idx, 1]  # col 1: linear velocity --> type string
        z = self.img_labels.iloc[idx, 2]  # col 2: angular velocity --> type string
        label = csv_format(x, z)  # type tensor 

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

    # LOAD MODEL
    learning_rate = 0.001
    weight_decay = 1e-5
    model = CNN(image_channels=3, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) 

    """
    NOTES:
    - Mean squared error is used to calculate the difference between prediction and label
    - Number of training epochs describes the number of times to train over entire dataset
    
    MSELoss() Example::

    >>> loss = nn.MSELoss()
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> target = torch.randn(3, 5)
    >>> output = loss(input, target)
    >>> output.backward()
    """

    # mean squared error class
    mse = nn.MSELoss()  # takes tensor as input
    num_training_epochs = 10 # loop through training data n times
    train_dataloader = DataLoader(training_data, batch_size=500, shuffle=True)  # load images/labels into 

    # TRAIN MODEL
    for epoch in range(num_training_epochs):  # loop through data 100 times
        data_iter = iter(train_dataloader)
        for train_features, train_labels in data_iter:

            """
            mse = nn.MSELoss()
            input: prediction = model(train_features) --> torch.size([64, 2])
            target: train_labels --> torch.size([64, 2])
            loss = mse(input, target)
            """

            # print(train_labels.size())  # output: torch.size([64, 2])

            prediction = model(train_features)  # output: torch.size([64, 2])
            print("prediction", prediction[0])
            print("labels", train_labels[0])

            # loss is a scalar
            loss = mse(prediction, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())

PATH = '/home/aj/models/loss_' + str(loss.item()) + '.pt'
torch.save(model.state_dict(), PATH)
print("------------")
print("MODEL SAVED!")
print("------------")