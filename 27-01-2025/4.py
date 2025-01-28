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
    def __init__(self):  # Fixed __init__
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
    def __init__(self):  # Fixed __init__
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

# Load the pre-trained model
def load_model():
    model_path = '/scratch/isl_77/ISL/Lab_4_3_model.pt'  # Replace with the path to the pre-trained model
    model = NN2()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"Model loaded from {model_path}")
    return model

# Freeze NN1 parameters
def freeze_nn1(model):
    for param in model.nn1.parameters():
        param.requires_grad = False
    print("NN1 parameters have been frozen.")

# Train the NN2 component with frozen NN1
def train(model, train_loader, criterion, optimizer):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# Evaluate the model
def evaluate(model, test_loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Main execution
if __name__ == "__main__":  # Fixed __name__
    # Load the pre-trained model
    model = load_model()

    # Freeze NN1 parameters
    freeze_nn1(model)

    # Define the loss function and optimizer for retraining NN2
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Retrain the model with the modified dataset
    train(model, train_loader, criterion, optimizer)

    # Evaluate the model on the test dataset
    evaluate(model, test_loader)
