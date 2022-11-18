import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from customize_dataset import CustomImageDataset
from model import CNN

training_data = CustomImageDataset()

# for i in range(training_data.__len__()):
        # print(training_data.__getitem__(i))

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
plot_loss = []
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
        print("loss:", loss.item())

        plot_loss.append(loss.item())

PATH = '/home/aj/catkin_ws/src/imitation_learning/models/loss_' + str(loss.item()) + '.pt'
# PATH = '/home/aj/models/loss_' + str(loss.item()) + '.pt'
torch.save(model.state_dict(), PATH)
print("------------")
print("MODEL SAVED!")
print("------------")

plt.plot(plot_loss)
plt.show()