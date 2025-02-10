import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 32  # Adjusted batch size
learning_rate = 0.001
num_epochs = 2  # Reduced epochs for faster training

# ImageNet transforms (smaller image size for faster training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to smaller size (128x128)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
data_path = '/scratch/data/imagenet-256/versions/1'
class_subset = list(range(10))  # Using first 10 classes

train_dataset = datasets.ImageFolder(root=data_path, transform=transform)
test_dataset = datasets.ImageFolder(root=data_path, transform=transform)

# Filter dataset to include only the first 10 classes
train_dataset.samples = [s for s in train_dataset.samples if s[1] in class_subset]
train_dataset.targets = [t for t in train_dataset.targets if t in class_subset]
test_dataset.samples = [s for s in test_dataset.samples if s[1] in class_subset]
test_dataset.targets = [t for t in test_dataset.targets if t in class_subset]

# DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Define CNN Model
class CNN(nn.Module):
    def __init__(self):  # Correct constructor name
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Apply MaxPooling to reduce size
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Apply MaxPooling to reduce size
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Apply MaxPooling to reduce size
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Apply MaxPooling to reduce size
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Apply MaxPooling to reduce size
        )
        # Adjusted size dynamically based on input
        self.fc = nn.Linear(256 * 4 * 4, 10)  # Adjust the size accordingly (256 * 4 * 4 after conv layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Initialize model, loss, and optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
def train():
    model.train()
    for epoch in range(num_epochs):
        total, correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')

# Testing the model
def test():
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Main execution
if __name__ == "__main__":
    train()
    test()
