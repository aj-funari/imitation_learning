import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image

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
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__ == "__main__":
    annotations_file = "/home/aj/images/labels/avoid_walls_labels.csv"
    img_dir = '/home/aj/images/avoid_walls/'
    data = CustomImageDataset(annotations_file, img_dir)
    print(data.__len__())
    print(data.__getitem__(-1))

    train_dataloader = DataLoader(data, batch_size=3, shuffle=True)
    print(train_dataloader)
    # test_dataloader = DataLoader(data, batch_size=64, shuffle=True)

