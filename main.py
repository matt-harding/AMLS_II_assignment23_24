import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch import save, load

class WhaleDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = self.data['image'].tolist()
        self.labels = self.data['individual_id'].tolist()
        self.classes = self.data['individual_id'].unique()
        self.encode = {k: i for i,k in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        image_name = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(image_name).convert('RGB')
        label = self.encode[self.labels[idx]]

        if self.transform:
            image = self.transform(image)

        return image, label
'''
    Whale Classifier neural network that inherits from PyTorch base neural network class
    Ref: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
'''
class WhaleClassifier(nn.Module):
    def __init__(self, num_classes):
        super(WhaleClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.embedding = nn.Embedding(num_classes, num_classes)
        
    def forward(self, x):
        # Convolutional layers
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)  # out.size() = (batch_size, 32, 128, 128)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)  # out.size() = (batch_size, 64, 64, 64)
        
        out = self.conv3(out)
        out = self.relu(out)     # out.size() = (batch_size, 128, 64, 64)
        
        # Flatten the output
        out = out.view(out.size(0), -1)
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out


if __name__ == "__main__":
    '''
        Step 1: Converts PIL image to a PyTorch tensor, scaling ther pixel values to the range [0,1]

        MATT to CONSIDER!
        Step 2: Normalizes the tensor by subtracting the mean and dividing by th e standard deviation. The provided mean and standard deviation values are typical for ImageNet data,
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])


    dataset = WhaleDataset(csv_file='train.csv', image_dir='Datasets/train_images', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_classes = dataset.data['individual_id'].nunique()
    model = WhaleClassifier(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch: {epoch+1}, Loss: {epoch_loss:.4f}')

    with open('model_state.pt', 'wb') as f:
        save(model.state_dict(), f)