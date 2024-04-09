class WhaleClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.2):
        super(WhaleClassifier, self).__init__()
        self.dropout_rate = dropout_rate

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # Added convolutional layer

        # Dropout layers for convolutional layers
        self.dropout_conv = nn.Dropout2d(self.dropout_rate)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 32 * 32, 512)  # Modified input size for fc1
        self.dropout_fc1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)

        # Is this needed if we're not doing inference?
        self.embedding = nn.Embedding(num_classes, num_classes)

    def forward(self, x):
        # Convolutional layers
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout_conv(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout_conv(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.dropout_conv(out)

        out = self.conv4(out)  # Added convolutional layer
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout_conv(out)  # Apply dropout to the new convolutional layer

        # Flatten the output
        out = out.view(out.size(0), -1)

        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout_fc1(out)
        out = self.fc2(out)

        return out