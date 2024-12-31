# Telco Customer Churn Prediction

This repository contains a project for predicting customer churn in a telecommunications company using various machine learning models.

## Overview

Customer churn prediction helps businesses identify customers who are likely to leave. This project involves data preprocessing, exploratory data analysis (EDA), handling class imbalance, feature scaling, and training multiple machine learning models to predict customer churn.

## Dataset

The dataset used in this project is the Telco Customer Churn dataset, which contains information about customer demographics, services subscribed, and account information.

## Project Structure

- `data/` - Contains the dataset (`Telco_customer_churn.csv`).
- `notebooks/` - Jupyter notebooks for exploration and model development.
- `src/` - Source code for data preprocessing, EDA, and model training.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/fatemi-loyalist/teleco_churn_prediction_project.git
    cd teleco_churn_prediction_project
    ```

2. Set up a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Data Preprocessing**:
    - The script preprocesses the data by handling missing values, encoding categorical variables, and scaling features.

2. **Exploratory Data Analysis (EDA)**:
    - Visualize data distributions and correlations.
    - Understand the relationships between features.

3. **Model Training**:
    - Train multiple machine learning models.
    - Evaluate model performance using accuracy, classification reports, and confusion matrices.

4. **Run the Script**:
    - Execute the main script to load data, preprocess, perform EDA, apply SMOTE, scale features, and train models:
    ```bash
    python src/main.py
    ```
