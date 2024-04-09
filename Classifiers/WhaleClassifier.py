import torch.nn as nn

"""
    Whale Classifier neural network that inherits from PyTorch base neural network class
    Ref: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
"""


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

    def forward(self, x):
        # Convolutional layers
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)  # out.size() = (batch_size, 32, 128, 128)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)  # out.size() = (batch_size, 64, 64, 64)

        out = self.conv3(out)
        out = self.relu(out)  # out.size() = (batch_size, 128, 64, 64)

        # Flatten the output
        out = out.view(out.size(0), -1)

        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out
