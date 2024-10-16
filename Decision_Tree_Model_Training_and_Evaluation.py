#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 22:16:19 2024
@author: jeongwoohong
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# Import machine learning libraries
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Load the preprocessed training and test datasets
train_data = pd.read_csv('/path/to/processed_large_train_data.csv')
test_data = pd.read_csv('/path/to/processed_test_data.csv')

# Create the binary target variable based on price
# 1 if price is <= 19,500, otherwise 0
train_data['target'] = (train_data['price'] <= 19500).astype(int)

#Decision Tree Model with Feature Importance Visualization

# Define the list of features to be used in the model
features = [
    'mileage', 'year', 'make_name', 'model_name', 'body_type',
    'engine_displacement', 'horsepower', 'fuel_type', 'transmission_display',
    'wheel_system', 'engine_cylinders', 'city_fuel_economy', 'highway_fuel_economy',
    'maximum_seating', 'length', 'width', 'height', 'wheelbase'
]

# Split features (X) and target variable (y) from the training data
X_train = train_data[features]
y_train = train_data['target']

# Initialize a Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree model on the training data
dt_model.fit(X_train, y_train)

# Evaluate the model using the training data itself (accuracy score)
y_train_pred = dt_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.3f}")

# Confusion Matrix: Visualize classification performance
conf_matrix = confusion_matrix(y_train, y_train_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Training Data)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot the feature importance determined by the Decision Tree model
feature_importance = dt_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance, color='skyblue')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
