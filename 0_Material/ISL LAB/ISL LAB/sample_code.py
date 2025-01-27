from sklearn.datasets import load_digits, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch import nn
import torch
import numpy as np

def get_data(type, split):
    if type == "scikit-minist-digits":
        data = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=float(split.split('-')[1])/100, random_state=42)
        return (X_train, y_train), (X_test, y_test)
    elif type == "scikit-20newsgroups":
        newsgroups_train = fetch_20newsgroups(subset='train')
        newsgroups_test = fetch_20newsgroups(subset='test')
        return (newsgroups_train.data, newsgroups_train.target), (newsgroups_test.data, newsgroups_test.target)
    else:
        raise ValueError("Invalid dataset type")

def get_model(type, config):
    if type == "randomforest":
        return RandomForestClassifier(n_estimators=config["num_estimators"], max_depth=config["depth"], class_weight='balanced')
    elif type == "nn":
        arch = config["arch_string"].split(',')
        layers = []
        for i in range(1, len(arch) - 1, 2):
            if arch[i].startswith('conv2d'):
                layers.append(nn.Conv2d(arch[i + 1][0], arch[i + 1][1], kernel_size=arch[i + 1][2], stride=arch[i + 1][3], padding=arch[i + 1][4]))
                layers.append(nn.ReLU())
            elif arch[i] == 'flatten_1':
                layers.append(nn.Flatten())
            elif arch[i].startswith('linear'):
                layers.append(nn.Linear(arch[i - 1], arch[i + 1]))
                if 'relu' in arch[i + 2]:
                    layers.append(nn.ReLU())
        layers.append(nn.Linear(arch[-2], arch[-1]))
        if 'sigmoid' in arch[-1]:
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)
    else:
        raise ValueError("Invalid model type")

def train_model(model, train_data, config=None):
    if isinstance(model, RandomForestClassifier):
        return model.fit(*train_data)
    elif isinstance(model, nn.Module):
        # Assuming PyTorch training
        X_train, y_train = train_data
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        epochs = 10 if config is None or 'epochs' not in config else config['epochs']
        batch_size = 16 if config is None or 'batch' not in config else config['batch']
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                inputs = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32)
                targets = torch.tensor(y_train[i:i+batch_size], dtype=torch.long)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        return model
    else:
        raise ValueError("Invalid model type")

def evaluate(model, metrics, test_data):
    X_test, y_test = test_data
    if isinstance(model, RandomForestClassifier):
        y_pred = model.predict(X_test)
        results = {}
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y_test, y_pred)
        if 'map' in metrics:
            results['map'] = average_precision_score(y_test, y_pred, average='macro')
        if 'avg_auc' in metrics:
            results['avg_auc'] = roc_auc_score(y_test, model.predict_proba(X_test), average='macro')
        return results
    elif isinstance(model, nn.Module):
        # Assuming PyTorch evaluation
        with torch.no_grad():
            inputs = torch.tensor(X_test, dtype=torch.float32)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            results = {}
            if 'accuracy' in metrics:
                results['accuracy'] = accuracy_score(y_test, predicted.numpy())
            # Implement other metrics if needed
            return results
    else:
        raise ValueError("Invalid model type")

def get_numclasses(data):
    if isinstance(data[1], np.ndarray):
        return len(np.unique(data[1]))
    else:
        return len(set(data[1]))

def get_dataitem(data, idx):
    return data[0][idx]

# Example usage
train_data, test_data = get_data(type="scikit-minist-digits", split="70-30")
model = get_model(type="randomforest", config={"num_estimators":150, "depth":10})
model = train_model(model=model, train_data=train_data)
metrics = evaluate(model=model, metrics=["accuracy", "map", "avg_auc"], test_data=test_data)
print(metrics)