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
batch_size = 32
learning_rate = 0.001
num_epochs = 2

# Transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
data_path = '/scratch/data/imagenet-256/versions/1'
class_subset = list(range(10))  # Using the first 10 classes

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

# Define NN1
class NN1(nn.Module):
    def __init__(self):  # Corrected __init__
        super(NN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        return x

# Define NN2
class NN2(nn.Module):
    def __init__(self):  # Corrected __init__
        super(NN2, self).__init__()
        self.nn1 = NN1()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 8 * 8, 10)

    def forward(self, x):
        x = self.nn1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Model, loss, optimizer
model = NN2().to(device)
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

# Save the model
def save_model():
    save_path = '/scratch/isl_78/ISL/Lab_4_3_model.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

# Load the model
def load_model():
    load_path = '/scratch/isl_78/ISL/Lab_4_3_model.pt'
    model.load_state_dict(torch.load(load_path))
    model.eval()
    print(f"Model loaded from {load_path}")

# Prediction
def predict():
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
    return all_predictions

# Main execution
if __name__ == "__main__":  # Corrected __name__
    train()
    test()
    save_model()
    load_model()
    predictions = predict()
    print(f"Predictions on test dataset: {predictions}")
