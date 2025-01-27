import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

def get_mnist_tensor():
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    X = torch.cat([mnist_train.data, mnist_test.data], dim=0).reshape(-1, 28 * 28).float()
    y = torch.cat([mnist_train.targets, mnist_test.targets], dim=0).long()

    return X, y

def get_mynn(inp_dim, hid_dim, num_classes):
    mynn = nn.Sequential(
        nn.Linear(inp_dim, hid_dim),
        nn.ReLU(),
        nn.Linear(hid_dim, num_classes)
    )
    return mynn

def train_combined_encdec_predictor(mynn, X, y, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mynn.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = mynn(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    return mynn

def get_loss_on_single_point(mynn, x0, y0):
    criterion = nn.CrossEntropyLoss()
    outputs = mynn(x0.unsqueeze(0))
    loss_val = criterion(outputs, torch.tensor([y0]))
    return loss_val
