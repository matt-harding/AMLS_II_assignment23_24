from PIL import Image
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch import save, load
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast

from Utils import WhaleDataset
from Classifiers import WhaleClassifier

BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 10

if __name__ == "__main__":
    # Load data
    dataset = WhaleDataset(csv_file="train.csv", image_dir="train_images")

    # Split data in train and test
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # For this assessment we only consider inviduals within the training dataset
    # This line would need to be modified to consider out of distribution
    num_classes = dataset.data["individual_id"].nunique()

    # Instantiate Image Classifier
    model = WhaleClassifier(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    """
        TRAIN CLASSIFIER
    """
    for epoch in range(EPOCHS):
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Add brief description
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Description on torch backprop implementation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}")

    # Need a dynamic model name linked to env file used
    with open("model_state.pt", "wb") as f:
        save(model.state_dict(), f)

    """
        TEST CLASSIFIER
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    # No Grad context manager used as no longer updating model parameters
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
