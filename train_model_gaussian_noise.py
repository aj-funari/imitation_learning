import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from customize_dataset import CustomImageDataset
from model import CNN
from helpers import save_to_csv_file

learning_rate = 0.001
weight_decay = 1e-5
model = CNN(image_channels=3, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

x = torch.randn(64, 3, 224, 224)
# print(x)
y = torch.ones(64, 2) * 5
# print(y)

mse = nn.MSELoss()
num_training_epochs = 100

plot_loss = []
for epoch in range(num_training_epochs):
    prediction = model(x)
    print("prediction:", prediction[0])

    loss = mse(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print("loss:", loss.item(), "\n")
    plot_loss.append(loss.item())

# print(plot_loss)
plt.plot(plot_loss)
plt.show()