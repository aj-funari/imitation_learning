import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__ == "__main__":
    dir = 'home/aj/images/avoid_walls/'
    for label in os.listdir(dir):
        label = label.split('-')

        if len(label) == 4:  # positive x and z coordinates
            x = label[0]
            z = label[1]
            action = [x,z]
        
        if len(label) == 5:  # negative x or z coordinate
            if label[0] == '': # -x
                x = '-' + label[1]
                z = label[2]
            if label[1] == '': # -z
                x = label[0]
                z = '-' + label[2]

        if len(label) == 6:  # negative x and z coordinates
            x = '-' + label[1]
            z = '-' + label[3]
        
        action = [x,z]

    annotations_file = "/home/aj/images/labels.csv"
    set = CustomImageDataset()