# Loan Grade Prediction Project

## Overview
This repository contains the code and documentation for a machine learning project focused on predicting loan grades for individuals based on historical loan data. The project aims to develop a reliable and accurate model that can generalize well to new, unseen data and contribute to making trustworthy predictions of loan grades.

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
   - [Data Preprocessing](#1-data-preprocessing)
   - [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
   - [Feature Engineering](#3-feature-engineering)
   - [Data Splitting](#4-data-splitting)
   - [Data Scaling](#5-data-scaling)
   - [Model Training and Evaluation](#6-model-training-and-evaluation)
   - [Prediction with RandomForestClassifier](#7-prediction-with-randomforestclassifier)

## 1. Data Preprocessing
- **Read the Data:** Load training and test datasets.
- **Data Inspection:** Examine the structure, summary statistics, and information of the datasets.
- **Handling Categorical Data:** Address variations in categorical columns, specifically in employment length.

## 2. Exploratory Data Analysis (EDA)
- **Categorical Variables Analysis:** Visualize count plots for categorical variables.
- **Categorical Variables vs. Loan Grade:** Analyze the distribution of categorical variables with respect to loan grades.
- **Numerical Variables Analysis:** Explore the distribution of numerical variables.
- **Correlation Analysis:** Examine correlation among numerical variables using a heatmap.
- **Scatter Plot Analysis:** Create scatter plots among numerical variables.

## 3. Feature Engineering
- **Handling Missing Values:** Identify and handle missing values in the datasets.
- **Handling Duplicates:** Check for and handle duplicate rows in the datasets.
- **Outlier Detection:** Identify and visualize outliers in numerical variables.
- **Encoding Categorical Variables:** Utilize one-hot encoding and ordinal encoding for categorical variables.
- **Mapping Variations:** Map variations in employment length and loan grade categories.

## 4. Data Splitting
- **Class Imbalance Handling:** Implement oversampling (SMOTE) and undersampling (RandomUnderSampler) techniques to address class imbalance.
- **Train-Test Split:** Split the data into training and test sets.

## 5. Data Scaling
- **Feature Scaling:** Use MinMaxScaler for scaling numerical features.

## 6. Model Training and Evaluation
- **Classifier Evaluation:** Train and evaluate various classifiers, including DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, KNeighborsClassifier, LogisticRegression, XGBClassifier, CatBoostClassifier, and LGBMClassifier.
- **Hyperparameter Tuning:** Perform hyperparameter tuning for DecisionTreeClassifier using GridSearchCV.

## 7. Prediction with RandomForestClassifier
- **Classifier Training:** Train the RandomForestClassifier with scaled training data.
- **Test Data Prediction:** Make predictions on the test data and map numeric labels to loan grade categories.
- **Result Submission:** Save the results to a CSV file for submission.
