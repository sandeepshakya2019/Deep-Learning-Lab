from sklearn.datasets import make_blobs, make_circles
import torch
from torchvision import datasets, transforms

def get_data_blobs(n_points):
    X, y = make_blobs(n_samples=n_points, centers=3, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y

def get_data_circles(n_points):
    X, y = make_circles(n_samples=n_points, noise=0.1, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y

def get_data_mnist():
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Combine train and test sets
    X = torch.cat([mnist_train.data, mnist_test.data], dim=0).float()
    y = torch.cat([mnist_train.targets, mnist_test.targets], dim=0).long()

    return X, y
