#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 22:16:19 2024
@author: jeongwoohong
"""

import pandas as pd
import numpy as np

# Load the raw training and test datasets
large_data = pd.read_csv('/path/to/large_train_data.csv')
small_data = pd.read_csv('/path/to/small_train_data.csv')
test_data = pd.read_csv('/path/to/test_data.csv')

# Function for processing the datasets (applicable to all training and test datasets)
def process_dataset(dataset):
    """
    This function preprocesses the dataset by handling missing values, 
    encoding categorical variables, and engineering features based on car options.
    """
    # Handle missing values in numeric columns using the mean value for imputation
    numeric_features = dataset.select_dtypes(include=[np.number]).columns
    for feature in numeric_features:
        dataset[feature].fillna(dataset[feature].mean(), inplace=True)

    # Handle missing values in categorical columns by replacing them with the most frequent value
    categorical_features = dataset.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        dataset[feature].fillna(dataset[feature].mode()[0], inplace=True)

    # Example of encoding: Convert categorical features to numeric via one-hot encoding
    dataset = pd.get_dummies(dataset, columns=['make_name', 'model_name', 'fuel_type', 'transmission_display', 'wheel_system'])

    # Feature Engineering: Extract information from car options column
    if 'major_options' in dataset.columns:
        # Convert the 'major_options' column from string to list format for easier processing
        dataset['major_options'] = dataset['major_options'].apply(lambda x: eval(x) if pd.notna(x) else [])
        
        # For each option, create a new column indicating its presence (binary feature)
        for option in options:
            dataset[option] = dataset['major_options'].apply(lambda x: 1 if option in x else 0)
        
        # Drop the 'major_options' column after extracting relevant information
        dataset.drop(columns=['major_options'], inplace=True)
    
    return dataset

# List of car options to be transformed into features
options = [
    "Adaptive Cruise Control",
    "Blind Spot Monitoring",
    "Heated Seats",
    "Navigation System",
    "Sunroof/Moonroof",
    "Leather Seats",
    "Technology Package",
    "Sport Package",
    "Premium Sound Package",
    "Luxury Package"
]

# Apply the processing function to the datasets
large_data = process_dataset(large_data)
small_data = process_dataset(small_data)
test_data = process_dataset(test_data)

# Save the processed datasets for use in model training
large_data.to_csv('/path/to/processed_large_train_data.csv', index=False)
small_data.to_csv('/path/to/processed_small_train_data.csv', index=False)
test_data.to_csv('/path/to/processed_test_data.csv', index=False)
