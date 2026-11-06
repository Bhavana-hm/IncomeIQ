import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

df = pd.read_csv("datase/adult.csv", na_values="?")

print(df.head())
print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Check income distribution
print("\nIncome Distribution:")
print(df["income"].value_counts())

# Remove duplicate records
df = df.drop_duplicates()

print("\nAfter removing duplicates:")
print("Dataset Shape:", df.shape)
print("Duplicate Rows:", df.duplicated().sum())
# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Total missing values
print("\nTotal Missing Values:")
print(df.isnull().sum().sum())
# Handle missing values
categorical_columns = df.select_dtypes(include="object").columns

df[categorical_columns] = df[categorical_columns].fillna("Unknown")

# Check missing values after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())

print("\nFinal Dataset Shape:")
print(df.shape)
# Separate input features and target
X = df.drop("income", axis=1)
y = df["income"]

print("\nInput Features Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nTarget Classes:")
print(y.unique())
# Convert categorical columns into numerical values
X = pd.get_dummies(X, drop_first=True, dtype=int)

print("\nAfter Encoding:")
print("Encoded Features Shape:", X.shape)

print("\nFirst 5 Encoded Rows:")
print(X.head())
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining and Testing Data:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train, y_train)

print("\nRandom Forest Model Training Completed!")
# Predict income using test data
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=">50K")
recall = recall_score(y_test, y_pred, pos_label=">50K")
f1 = f1_score(y_test, y_pred, pos_label=">50K")

# Display results
print("\nMODEL EVALUATION RESULTS")
print("Accuracy:", accuracy * 100, "%")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Save the trained Random Forest model
joblib.dump(rf_model, "models/random_forest_model.pkl")

# Save feature column names for website predictions
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

print("\nModel and feature columns saved successfully!")