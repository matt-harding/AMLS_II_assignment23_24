from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch import save, load
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast

from Utils import WhaleDataset 
from Classifiers import WhaleClassifier 

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


    dataset = WhaleDataset(csv_file='Datasets/train.csv', image_dir='Datasets/train_images', transform=transform)
    train_data, test_data = train_test_split(dataset, test_size=0.95, random_state=42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # For this assessment we only considfer inviduals within the training dataset
    # This line would need to be modified to consider out of 
    num_classes = dataset.data['individual_id'].nunique()

    model = WhaleClassifier(num_classes)

    # Why are we using Cross Entropy?

    # Can we switch this out to 
    criterion = nn.CrossEntropyLoss()

    # Why are we using ADAM?
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        train_loss = 0.0
        batch = 1

        for inputs, labels in train_loader:
            #Remove logging before submitting
            print(f"Batch {batch} of {len(train_loader)}")
            inputs, labels = inputs.to(device), labels.to(device)
            # Add brief description
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Description on torch backprop implementation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch += 1

        train_loss /= len(train_loader)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}')

    # Need a dynamic model name linked to env file used
    with open('model_state.pt', 'wb') as f:
        save(model.state_dict(), f)


    # Break out testing into seperate file?
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    #Need to ecxplain this login in more detail
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
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')