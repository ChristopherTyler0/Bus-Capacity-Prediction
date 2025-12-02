# data analytics packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ML and AI packages: sklearn and torch
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, random_split

# system packages
import os
from PIL import Image

#pytorch dataset class
class busDataset(Dataset):
    def __init__(self, x, y):
        self.X = torch.tensor(x.values, dtype = torch.float32)
        self.Y = torch.tensor(y.values, dtype = torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

#read input data from csv file and select x and y columns 
df = pd.read_csv("busdata.csv")
featureCols = ['busNumber', 'stopID', 'prevCapacity', 'oncomers']
targetCols = ['actualCapacity']
x = df[featureCols]
y = df[targetCols]

#create dataset object, split, and shuffle
dataset = busDataset(x,y)
trainSize = int(0.8 * len(dataset))
testSize = len(dataset) - trainSize
trainDataset, testDataset = random_split(dataset, [trainSize, testSize])
trainLoader = DataLoader(trainDataset, batch_size = 32, shuffle = True)
testLoader = DataLoader(testDataset, batch_size = 32, shuffle = True)

#defining the pytorch model, layers come from torch module nn.
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 16, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(16,32,kernel_size=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = CNN()

#training
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
epochs = 50
testLosses = []
trainLosses = []
for epoch in range(epochs):
    trainLoss = 0.0
    #training phase
    model.train()
    for xBatch, yBatch in trainLoader:
        optimizer.zero_grad()
        output = model(xBatch)
        loss = criterion(output, yBatch)
        loss.backward()
        optimizer.step()
        trainLoss += loss.item()
    #test phase
    testLoss = 0.0
    with torch.no_grad():
        for xBatch, yBatch in testLoader:
            output = model(xBatch)
            loss = criterion(output, yBatch)
            testLoss += loss.item()
    testLosses.append(testLoss)
    trainLosses.append(trainLoss)
    print(f'Epoch {epoch+1}/{epochs}, Training Loss: {trainLoss:.4f} | Testing Loss: {testLoss:.4f}')



def plotlosses():
    # Plot the loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(trainLosses, label='Training Loss', color='red')
    plt.plot(testLosses, label='Testing Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Bus Prediction Training and Testing Loss Curves')
    plt.legend()
    plt.savefig(os.path.join('loss_curve.png'))  # Save the loss curve
    plt.show()

model.eval()
def getPrediction(busNumber, stopID, previousCapacity, oncomers):
    model.eval()
    example = torch.tensor([[busNumber, stopID, previousCapacity, oncomers]],dtype = torch.float32)
    with torch.no_grad():
        output = model(example)
    return output.item()

total = 0
correct = 0
marginOfError = 0.75
for index, row in df.iterrows():
    predictedCapacity = getPrediction(row['busNumber'], row['stopID'], row['prevCapacity'], row['oncomers'])
    actualCapacity = row['actualCapacity']
    print(f"Stop Name: {row['stopName']} | Actual Capacity: {row['actualCapacity']}| Predicted Capacity: {predictedCapacity:.0f}")
    total += 1
    if (predictedCapacity >= actualCapacity * marginOfError) and (predictedCapacity <= actualCapacity / marginOfError):
        correct += 1
print(f"Final accuracy with margin of error {marginOfError}: {correct/total}")

