# -*- coding: utf-8 -*-
"""DT_generated.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SEMM83q3Tyo9n36KHUBPHb7AGACLpxsN

# Decision Tree classification on make moons, blobs and circles data set
"""

from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, average_precision_score

"""##Depth param = 1"""

def train_test_decision_tree(dataset_generator):
    X, y = dataset_generator()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth = 1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    classification_rep = classification_report(y_test, y_pred)


    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Classification Report:")
    print(classification_rep)

    classes = np.unique(y_test)
    mean_avg_precision = 0

    for class_label in classes:

        binary_y_test = (y_test == class_label).astype(int)
        binary_y_pred = (y_pred == class_label).astype(int)

        avg_precision = average_precision_score(binary_y_test, binary_y_pred)
        print(f"Average Precision for Class {class_label}: {avg_precision:.4f}")

        mean_avg_precision += avg_precision

    mean_avg_precision /= 3
    print(f"\nMean Average Precision: {mean_avg_precision:.4f}")

print("Decision Tree Classifier for make_moons dataset:")
train_test_decision_tree(make_moons)

print("\nDecision Tree Classifier for make_blobs dataset:")
train_test_decision_tree(make_blobs)

print("\nDecision Tree Classifier for make_circles dataset:")
train_test_decision_tree(make_circles)

"""##Depth Param = 2"""

def train_test_decision_tree(dataset_generator):
    X, y = dataset_generator()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth = 2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    classification_rep = classification_report(y_test, y_pred)


    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Classification Report:")
    print(classification_rep)

    classes = np.unique(y_test)
    mean_avg_precision = 0

    for class_label in classes:

        binary_y_test = (y_test == class_label).astype(int)
        binary_y_pred = (y_pred == class_label).astype(int)

        avg_precision = average_precision_score(binary_y_test, binary_y_pred)
        print(f"Average Precision for Class {class_label}: {avg_precision:.4f}")

        mean_avg_precision += avg_precision

    mean_avg_precision /= 3
    print(f"\nMean Average Precision: {mean_avg_precision:.4f}")

print("Decision Tree Classifier for make_moons dataset:")
train_test_decision_tree(make_moons)

print("\nDecision Tree Classifier for make_blobs dataset:")
train_test_decision_tree(make_blobs)

print("\nDecision Tree Classifier for make_circles dataset:")
train_test_decision_tree(make_circles)

"""##Depth param not mentioned"""

def train_test_decision_tree(dataset_generator):
    X, y = dataset_generator()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    classification_rep = classification_report(y_test, y_pred)


    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Classification Report:")
    print(classification_rep)

    classes = np.unique(y_test)
    mean_avg_precision = 0

    for class_label in classes:

        binary_y_test = (y_test == class_label).astype(int)
        binary_y_pred = (y_pred == class_label).astype(int)

        avg_precision = average_precision_score(binary_y_test, binary_y_pred)
        print(f"Average Precision for Class {class_label}: {avg_precision:.4f}")

        mean_avg_precision += avg_precision

    mean_avg_precision /= 3
    print(f"\nMean Average Precision: {mean_avg_precision:.4f}")

print("Decision Tree Classifier for make_moons dataset:")
train_test_decision_tree(make_moons)

print("\nDecision Tree Classifier for make_blobs dataset:")
train_test_decision_tree(make_blobs)

print("\nDecision Tree Classifier for make_circles dataset:")
train_test_decision_tree(make_circles)

