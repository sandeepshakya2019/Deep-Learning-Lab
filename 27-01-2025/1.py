import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader, random_split, Subset 
 
# Device configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
 
# Hyperparameters 
batch_size = 16  # Decreasing batch size to reduce memory usage
learning_rate = 0.001 
epochs = 5 
 
# Transform and load ImageNet dataset 
transform = transforms.Compose([ 
    transforms.Resize((128, 128)),  # Reducing image size for memory efficiency
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)) 
]) 
 
# Load dataset 
dataset = datasets.ImageFolder(root="/scratch/data/imagenet-256/versions/1", transform=transform) 
 
print("dataset size", len(dataset)) 
 
# Split dataset into training and testing sets 
train_size = int(0.8 * len(dataset)) 
test_size = len(dataset) - train_size 
train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) 
 
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) 
 
# Define the neural network 
class CNN(nn.Module): 
    def __init__(self): 
        super(CNN, self).__init__() 
        self.conv_layers = nn.Sequential( 
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2) 
        ) 
        # Reduced the fully connected layer size to prevent excessive memory usage
        self.fc = nn.Linear(256 * 16 * 16, 2)  # Adjusted for smaller image size 
 
    def forward(self, x): 
        x = self.conv_layers(x) 
        x = x.view(x.size(0), -1) 
        x = self.fc(x) 
        return x 
 
model = CNN().to(device) 
 
# Loss and optimizer 
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
 
# Training loop 
def train(): 
    model.train() 
    for epoch in range(epochs): 
        running_loss = 0.0 
        correct = 0 
        total = 0 
        for images, labels in train_loader: 
            images, labels = images.to(device), labels.to(device) 
 
            # Forward pass 
            outputs = model(images) 
            loss = criterion(outputs, labels) 
 
            # Backward pass 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
 
            running_loss += loss.item() 
            _, predicted = outputs.max(1) 
            total += labels.size(0) 
            correct += predicted.eq(labels).sum().item() 
 
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, " 
              f"Accuracy: {100 * correct / total:.2f}%") 
 
# Testing loop 
def test(): 
    model.eval() 
    correct = 0 
    total = 0 
    with torch.no_grad(): 
        for images, labels in test_loader: 
            images, labels = images.to(device), labels.to(device) 
            outputs = model(images) 
            _, predicted = outputs.max(1) 
            total += labels.size(0) 
            correct += predicted.eq(labels).sum().item() 
 
    print(f"Test Accuracy: {100 * correct / total:.2f}%") 
 
# Execute training, testing, and visualization 
train() 
test() 
