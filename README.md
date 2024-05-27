# Loan Grade Prediction

## Table of Contents

1. [Overview](#overview)
2. [Libraries Used](#libraries-used)
3. [Data Loading](#data-loading)
4. [Data Preprocessing](#data-preprocessing)
    - [Handling Categorical Variables](#handling-categorical-variables)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Feature Engineering](#feature-engineering)
    - [Numerical Variable Transformation](#numerical-variable-transformation)
    - [Categorical Variable Encoding](#categorical-variable-encoding)
6. [Data Splitting](#data-splitting)
7. [Model Training and Evaluation](#model-training-and-evaluation)
    - [Decision Tree, Random Forest, XGBoost](#decision-tree-random-forest-xgboost)
    - [Feature Importance with RFECV](#feature-importance-with-rfecv)
    - [Hyperparameter Tuning and StratifiedKFold Validation](#hyperparameter-tuning-and-stratifiedkfold-validation)
8. [Model Prediction](#model-prediction)
    - [Decision Tree Classifier](#decision-tree-classifier)
9. [Submission](#submission)
    - [Result CSV File](#result-csv-file)
10. [Instructions for Running the Code](#instructions-for-running-the-code)
11. [Note](#note)

## Overview:

The project aims to predict loan grades for individuals using historical loan data. Key steps include data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## Libraries Used:

- `numpy`: Numerical operations.
- `pandas`: Data manipulation and analysis.
- `matplotlib` and `seaborn`: Data visualization.
- `sklearn`: Machine learning tasks (preprocessing, model selection, evaluation).
- `imblearn`: Handling imbalanced datasets.
- `catboost`, `xgboost`, `lightgbm`: Gradient boosting libraries.

## Data Loading:

- Training data loaded from "train.csv" into 'train' DataFrame.
- Test data loaded from "test.csv" into 'test' DataFrame.

## Data Preprocessing:

### 1. Handling Categorical Variables:
- Identified and processed categorical columns.
- Standardized values in "근로기간" column.

### 2. Exploratory Data Analysis (EDA):
- Visualized distribution of categorical variables.
- Count plots show the relationship between categorical variables and loan grades.
- Explored distribution and correlation of numerical variables.

## Feature Engineering:

### 1. Numerical Variable Transformation:
- Addressed missing values, duplicates, and outliers.
- Created new features related to loan repayment ratios, income, etc.

### 2. Categorical Variable Encoding:
- Applied label encoding to categorical variables.
- Performed ordinal encoding on "대출등급" column.

## Data Splitting:

- Data split into features (X) and target variable (y) using `train_test_split`.
- Created training and validation sets.

## Model Training and Evaluation:

### 1. Decision Tree, Random Forest, XGBoost:
- Trained and evaluated Decision Tree, Random Forest, and XGBoost classifiers.
- Used F1 scores as the evaluation metric.

### 2. Feature Importance with RFECV:
- Utilized Recursive Feature Elimination with Cross-Validation (RFECV) for feature selection.
- Visualized feature importances for Decision Tree.

### 3. Hyperparameter Tuning and StratifiedKFold Validation:
- Conducted grid search for hyperparameter tuning.
- Used Stratified K-Fold cross-validation for model evaluation.

## Model Prediction:

### 1. Decision Tree Classifier:
- Selected the best Decision Tree model.
- Feature selection based on RFECV results.
- Trained the model on the entire dataset.
- Made predictions on the test dataset.

## Submission:

### 1. Result CSV File:
- Mapped predicted loan grades back to original categories.
- Created a DataFrame with "ID" and "대출등급" columns.
- Saved DataFrame as "result_submission.csv" for submission.

## Instructions for Running the Code:

1. Ensure all necessary libraries are installed (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `catboost`, `xgboost`, `lightgbm`).
2. Place the training data in a file named "train.csv" and the test data in a file named "test.csv".
3. Run the code script, and it will perform data preprocessing, EDA, feature engineering, model training, and prediction.
4. Final predictions will be saved in a CSV file named "result_submission.csv".

## Note:

- The code assumes availability of training and test datasets in CSV format.
- Provides flexibility for using different classifiers and handling imbalanced datasets.
- Hyperparameters and features can be fine-tuned based on specific requirements.

