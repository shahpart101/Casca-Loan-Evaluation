import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import joblib

# Load the first two CSV files
def load_data():
    df1 = pd.read_csv('sample1.csv')
    df2 = pd.read_csv('sample 2.csv')
    return pd.concat([df1, df2], ignore_index=True)

# Feature engineering for CFRS and Interest Rate models
def feature_engineering(df):
    df['Net Flow'] = df['Credit'].fillna(0) - df['Debit'].fillna(0)
    df['PositiveBalance'] = df['Balance'] > 0
    
    monthly_summary = df.groupby(pd.Grouper(key='Date', freq='M')).agg({
        'Credit': 'sum',
        'Debit': 'sum',
        'Net Flow': 'sum',
        'PositiveBalance': 'mean'
    }).reset_index()

    monthly_summary['IncomeStability'] = monthly_summary['Credit'].rolling(3).std().fillna(0)
    monthly_summary['ExpenseStability'] = monthly_summary['Debit'].rolling(3).std().fillna(0)
    
    return monthly_summary

# CFRS Model
def train_cfrs_model(df):
    df['HighRisk'] = (df['Net Flow'] < 0).astype(int)
    X = df[['Credit', 'Debit', 'Net Flow', 'IncomeStability', 'ExpenseStability', 'PositiveBalance']]
    y = df['HighRisk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\nCFRS Model Classification Report:\n", classification_report(y_test, y_pred))
    
    joblib.dump(model, 'models/cfrs_model.pkl')
    return model

# Interest Rate Model
def train_interest_rate_model(df):
    df['InterestRate'] = 5 + (df['IncomeStability'] * 0.1) + (df['ExpenseStability'] * 0.1)
    X = df[['Credit', 'Debit', 'Net Flow', 'IncomeStability', 'ExpenseStability']]
    y = df['InterestRate']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\nInterest Rate Model MSE:", mean_squared_error(y_test, y_pred))
    
    joblib.dump(model, 'models/interest_rate_model.pkl')
    return model

# Main function to run both models
def main():
    df = load_data()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = feature_engineering(df)
    
    print("Training CFRS Model...")
    train_cfrs_model(df)
    
    print("Training Interest Rate Model...")
    train_interest_rate_model(df)
    
    print("\nModels saved successfully!")

if __name__ == "__main__":
    main()
