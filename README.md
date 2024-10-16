# Car_Price_Classification_Competition
A machine learning project for a classification competition, using a Decision Tree algorithm to predict whether the price of a used car is below or above $19,500. The project emphasizes feature engineering to improve prediction accuracy and achieve an overall model accuracy of 0.917. 

## Overview
This repository is part of a classification competition aimed at predicting whether the price of a used car is below or above $19,500. The model uses a **Decision Tree** algorithm, with a strong focus on **feature engineering** to enhance accuracy. The final model achieved an accuracy of **0.917** on the training set.

## Project Structure
The project files are organized as follows:

- **`car_price_classification.py`**: Python script containing the code for data preprocessing, feature engineering, model training, and evaluation.
- **`submission.csv`**: CSV file containing predictions for the test set, formatted according to the competition guidelines.
- **`report.pdf`**: PDF file containing:
  - Anonymized name (e.g., BellKor97).
  - Prediction accuracy on the training set.
  - Confusion matrix for the training data.
  - Graphs showing feature importance, KNN estimation patterns, and error rates as per the assignment requirements.

## Methodology
### 1. Data Preprocessing
- Handled missing values by imputing with mean/median values for numeric features.
- Normalized and scaled relevant features for consistency.

### 2. Feature Engineering
- Constructed new features using domain knowledge to improve the predictive power of the model.
- Selected the most relevant features to enhance model accuracy.

### 3. Model Training
- Implemented a **Decision Tree** algorithm for binary classification (predicting whether the price is ≤ $19,500 or > $19,500).
- Applied cross-validation to evaluate and optimize the model's performance.

### 4. Evaluation Metrics
- **Accuracy**: 0.917
- **Confusion Matrix**: Included in the report to visualize classification results.

### 5. Feature Importance
- Analyzed and visualized the importance of each feature in determining car prices using the Decision Tree model.

### 6. KNN and Error Analysis
- Replicated patterns from ISLR figure 2.17 and figure 4.7, demonstrating the behavior of KNN and error rates based on model flexibility and classification thresholds.
- Generated ROC curves to visualize model performance.
