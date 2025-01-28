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

class NN1(nn.Module):
    def _init_(self):
        super(NN1, self)._init_()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 3 input channels (RGB) -> 16 output channels
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # MaxPooling to reduce size
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16 input channels -> 32 output channels
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # MaxPooling to reduce size

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)  # Max pooling after first block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)  # Max pooling after second block
        return x


class NN2(nn.Module):
    def _init_(self):
        super(NN2, self)._init_()
        self.nn1 = NN1()  # Use NN1 as part of the model
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 input channels -> 64 output channels
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)  # MaxPooling to reduce size
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64 input channels -> 128 output channels
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)  # MaxPooling to reduce size
        self.fc = nn.Linear(128 * 8 * 8, 10)  # Fully connected layer (adjust size based on output of convolutions)

    def forward(self, x):
        x = self.nn1(x)  # Pass input through NN1
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)  # Max pooling after third block
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)  # Max pooling after fourth block
        x = x.view(x.size(0), -1)  # Flatten the tensor before passing to the fully connected layer
        x = self.fc(x)
        return x


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

# Visualize a sample
def visualize_sample():
    model.eval()
    sample, label = next(iter(test_loader))
    sample, label = sample.to(device), label.to(device)
    output = model(sample)
    _, predicted = torch.max(output, 1)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    unnormalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    
    sample_img = unnormalize(sample[0].cpu())
    plt.imshow(sample_img.permute(1, 2, 0).clamp(0, 1).numpy())
    plt.title(f"True: {label[0].item()}, Predicted: {predicted[0].item()}")
    plt.show()

# Main execution
if _name_ == "_main_":
    train()
    test()
    visualize_sample()