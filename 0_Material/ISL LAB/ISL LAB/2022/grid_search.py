import torch
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from torchvision import datasets, transforms

def get_data_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Combine train and test sets
    X = torch.cat([mnist_train.data, mnist_test.data], dim=0).reshape(-1, 28 * 28).float()
    y = torch.cat([mnist_train.targets, mnist_test.targets], dim=0).long()

    return X, y

def build_lr_model(X, y):
    clf_lr = LogisticRegression(max_iter=1000)
    clf_lr.fit(X.numpy(), y.numpy())
    return clf_lr

def build_rf_model(X, y):
    clf_rf = RandomForestClassifier()
    clf_rf.fit(X.numpy(), y.numpy())
    return clf_rf

def get_paramgrid_lr():
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    return param_grid

def get_paramgrid_rf():
    param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [5, 10, 15]}
    return param_grid

def perform_gridSearch_cv_multimetric(model1, param_grid, X, y, metrics):
    grid_search = GridSearchCV(model1, param_grid, cv=5, scoring=metrics, refit=False)
    grid_search.fit(X.numpy(), y.numpy())

    top_scores = []
    for metric in metrics:
        best_index = np.argmax(grid_search.cv_results_['mean_test_' + metric])
        top_scores.append(grid_search.cv_results_['mean_test_' + metric][best_index])

    return top_scores
