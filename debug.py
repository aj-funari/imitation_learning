import os
import cv2
import torch
from model import CNN

### LOAD MODEL
model = CNN(image_channels=3, num_classes=2)

DATADIR = '/home/aj/images/avoid_walls'
for label in os.listdir(DATADIR):
    img_array = cv2.imread(os.path.join(DATADIR, label))  # <type 'numpy.ndarray'>
    img_tensor = torch.from_numpy(img_array).float()  # <class 'torch.Tensor')
    tensor = img_tensor.reshape([1, 3, 224, 224])
    # print(img_tensor)

    print("Image label:", label)
    prediction = model(tensor)
    print("Model prediction:", prediction)


print("-------------------------------------")

# LOAD TRAINED MODEL
model = CNN(image_channels=3, num_classes=2)
PATH = '/home/aj/models/loss_0.34640446305274963.pt'
model.load_state_dict(torch.load(PATH))
model.eval()

DATADIR = '/home/aj/images/avoid_walls'
for label in os.listdir(DATADIR):
    img_array = cv2.imread(os.path.join(DATADIR, label))  # <type 'numpy.ndarray'>
    img_tensor = torch.from_numpy(img_array).float()  # <class 'torch.Tensor')
    tensor = img_tensor.reshape([1, 3, 224, 224])
    # print(img_tensor)

    print("Image label:", label)
    prediction = model(tensor)
    print("Model prediction:", prediction)