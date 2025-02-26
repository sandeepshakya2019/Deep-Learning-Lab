import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn

# Define dataset path
data_path = '/scratch/data/imagenet-256/versions/1'

# Define transformations
transform = transforms.ToTensor()

# Load full dataset
full_dataset = datasets.ImageFolder(root=data_path, transform=transform)

# Get class-to-index mapping and sort class names
class_to_idx = full_dataset.class_to_idx
sorted_classes = sorted(class_to_idx.keys())  # Sort class names alphabetically
top_3_classes = sorted_classes[:10]  # Take the first 3 classes
top_3_indices = [class_to_idx[c] for c in top_3_classes]

print(f"Top 3 Selected Classes: {top_3_classes}")

# Filter dataset to keep only selected classes
filtered_samples = [s for s in full_dataset.samples if s[1] in top_3_indices]

# Create a new dataset with only the selected classes
full_dataset.samples = filtered_samples
full_dataset.targets = [s[1] for s in filtered_samples]

# Update class names
class_names = top_3_classes

print(f"Filtered Dataset Size: {len(full_dataset)}")
print(f"Total Classes After Filtering: {len(class_names)}")

# Split into train (80%) and test (20%) sets
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
print(f"Train Data size: {len(train_dataset)}, Test Data Size: {len(test_dataset)}")

# Create DataLoaders
train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

print(f"Train Batches: {len(train_dataloader)}, Test Batches: {len(test_dataloader)}")

# # Save the image instead of showing it
# for img, label in train_dataset:
#     print(f"Image shape: {img.shape}")
#     image_np = img.permute(1, 2, 0).numpy()
#     plt.imshow(image_np)
#     plt.title(class_names[label])
#     plt.axis(False)
    
#     # Save the image to a file
#     plt.savefig("sample_image.png", bbox_inches="tight")
#     print("Image saved as 'sample_image.png'. Download and view it locally.")
#     break  # Show only one image


# Define CNN Model
class CNNModel(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x

# Initialize model
model = CNNModel(3, 20, len(class_names))
dummy_image = torch.rand((32, 3, 256, 256))  # Batch of 32 images
print("Dummy Image shape:", dummy_image.shape)
model(dummy_image)

# Training setup
epochs = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Accuracy function
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc

# Training Loop
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")  
    train_loss = 0.0
    train_acc = 0.0
    batch_interval = 100  

    for batch_idx, (img, label) in enumerate(train_dataloader):
        model.train()
        y_pred = model(img)
        loss = loss_fn(y_pred, label)
        train_loss += loss.item()
        train_acc += accuracy_fn(y_pred.argmax(dim=1), label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % batch_interval == 0:
            print(f"  Batch {batch_idx+1}/{len(train_dataloader)} - Loss: {loss.item():.4f}")

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    
    print(f"Epoch {epoch+1} Summary -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n")

# **Test Loop**
model.eval()
test_loss = 0.0
test_acc = 0.0

with torch.no_grad():
    for img, label in test_dataloader:
        y_pred = model(img)
        loss = loss_fn(y_pred, label)
        test_loss += loss.item()
        test_acc += accuracy_fn(y_pred.argmax(dim=1), label)

test_loss /= len(test_dataloader)
test_acc /= len(test_dataloader)

print(f"Test Summary -> Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

count = 0
for img, label in test_dataset:
    # print("Using image : ", img.shape)
    y_pred = model(img.unsqueeze(0))
    image_np = img.permute(1, 2, 0).numpy()
    title = class_names[label] + " | " +  class_names[y_pred.argmax(dim=1)]
    plt.title(title)
    plt.axis(False)
    filename = str(count) + ".png"
    plt.imshow(image_np)
    plt.savefig(filename, bbox_inches="tight")
    count+=1
    if(count > 10) : break
print("Images Completed")