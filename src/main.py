import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")


# 1. Load Data
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data


# 2. Preprocess Data
def preprocess_data(data, columns_to_drop, missing_data_col):
    # Drop unnecessary columns
    data = data.drop(columns=columns_to_drop)

    # Handle missing values
    data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors='coerce')
    for col in missing_data_col:
        data.loc[(data[col].isna()) & (data['Tenure Months'] == 0), col] = 0

    # Convert specific categorical columns to numerical via mapping
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    binary_cols = ['Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Paperless Billing']
    for col in binary_cols:
        data[col] = data[col].map({'Yes': 1, 'No': 0})

    # Encode remaining categorical columns using LabelEncoder
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le  # Save encoders if needed for inverse transformation later

    return data


# 3. Exploratory Data Analysis
def plot_distributions(data, numerical_cols):
    num_cols = len(numerical_cols)
    num_rows = (num_cols + 3) // 4  # Calculate the number of rows needed (4 columns per row)

    plt.figure(figsize=(15, num_rows * 5))  # Adjust figure size based on the number of rows
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(num_rows, 4, i)  # Dynamically calculate the position of the subplot
        sns.histplot(data[col], bins=30, color='blue', kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()


# 4. Feature Selection
def feature_selection(data, numerical_cols):
    # Correlation heatmap
    corr_matrix = data[numerical_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()


# 5. Handle Imbalance with SMOTE
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_multiple_models(models, X_train, y_train, X_test, y_test):
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = accuracy
        print_model_performance(model_name, y_test, y_pred)
    print_summary(results)


def print_model_performance(model_name, y_test, y_pred):
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("-" * 60)


def print_summary(results):
    print("\nSummary of Accuracies:")
    for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name}: {accuracy:.2f}")


# Train multiple models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(probability=True, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Naive Bayes": GaussianNB()
}

# Main Execution
if __name__ == "__main__":
    # Filepath
    filepath = "data/Telco_customer_churn.csv"

    # Columns to drop and missing data columns
    columns_to_drop = ['CustomerID', 'Count', 'Country', 'State', 'Lat Long', 'Zip Code', 'Churn Label', 'City',
                       'Latitude', 'Longitude', 'Churn Score', 'Churn Reason']
    missing_data_col = ['Total Charges']

    # Load and preprocess data
    data = load_data(filepath)
    data = preprocess_data(data, columns_to_drop, missing_data_col)

    # Check for numerical columns before plotting distributions
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    print("Numerical Columns:", numerical_cols)  # Debugging

    if numerical_cols:
        plot_distributions(data, numerical_cols)  # Call for EDA
    else:
        print("No numerical columns available for distribution plots.")

    # Call feature_selection if numerical columns exist
    if numerical_cols:
        feature_selection(data, numerical_cols)
    else:
        print("No numerical columns available for correlation heatmap.")

    # Split into features and target
    X = data.drop('Churn Value', axis=1)
    y = data['Churn Value']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale data
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Apply SMOTE
    X_train_smote, y_train_smote = apply_smote(X_train_scaled, y_train)

    # Train and evaluate models
    results = train_multiple_models(models, X_train_smote, y_train_smote, X_test_scaled, y_test)