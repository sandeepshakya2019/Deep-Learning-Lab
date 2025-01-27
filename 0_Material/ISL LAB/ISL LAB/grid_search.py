from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Define the decision tree classifier
dt_classifier = DecisionTreeClassifier()

# Define the parameter grid
param_grid = {
    'max_depth': [1, 3, 5, None],
    'criterion': ['gini', 'entropy']
}

# Perform grid search cross-validation
grid_search = GridSearchCV(dt_classifier, param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Report best parameters
print("Best Parameters:", grid_search.best_params_)

# Compute accuracy on the test set
best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Set:", accuracy)