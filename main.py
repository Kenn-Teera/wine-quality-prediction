import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Fetch dataset 
wine_quality = fetch_ucirepo(id=45) 
X = wine_quality.data.features 
y = wine_quality.data.targets 

# Basic data analysis
def analyze_data(X, y):
    print("\n=== Data Analysis ===")
    print(f"Dataset shape: {X.shape}")
    print("\nMissing values:")
    print(X.isnull().sum())
    print("\nFeature statistics:")
    print(X.describe())

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()

# Preprocess data
def preprocess_data(X, y):
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values.ravel(), 
        test_size=0.2, 
        random_state=42
    )
    return X_train, X_test, y_train, y_test

# Train and evaluate model
def train_evaluate_model(X_train, X_test, y_train, y_test):
    # Train Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Print model performance
    print("\n=== Model Performance ===")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance plot
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    return rf_model

def main():
    # Analyze data
    analyze_data(X, y)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Train and evaluate model
    model = train_evaluate_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()