import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import joblib
import os

# Load Data from CSVs
def load_data():
    df1 = pd.read_csv('sample1.csv')
    df2 = pd.read_csv('sample 2.csv')
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Convert columns to numeric, replacing non-numeric with NaN
    for col in ['Debit', 'Credit', 'Balance']:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)
    
    return combined_df

# Feature Engineering
def feature_engineering(df):
    df['Net Flow'] = df['Credit'] - df['Debit']
    df['PositiveBalance'] = df['Balance'] > 0

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    monthly_summary = df.groupby(pd.Grouper(key='Date', freq='M')).agg({
        'Credit': 'sum',
        'Debit': 'sum',
        'Net Flow': 'sum',
        'PositiveBalance': 'mean'
    }).reset_index()

    monthly_summary['IncomeStability'] = monthly_summary['Credit'].rolling(3).std().fillna(0)
    monthly_summary['ExpenseStability'] = monthly_summary['Debit'].rolling(3).std().fillna(0)

    return monthly_summary

# CFRS Model Training
def train_cfrs_model(df):
    df['HighRisk'] = (df['Net Flow'] < 0).astype(int)
    X = df[['Credit', 'Debit', 'Net Flow', 'IncomeStability', 'ExpenseStability', 'PositiveBalance']]
    y = df['HighRisk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nCFRS Model Classification Report:\n", classification_report(y_test, y_pred))

    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, 'models/cash_flow_risk_model.pkl')  # Save as cash_flow_risk_model.pkl to match API

    return model

# Interest Rate Model Training
def train_interest_rate_model(df):
    df['InterestRate'] = 5 + (df['IncomeStability'] * 0.1) + (df['ExpenseStability'] * 0.1)
    X = df[['Credit', 'Debit', 'Net Flow', 'IncomeStability', 'ExpenseStability']]
    y = df['InterestRate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nInterest Rate Model Mean Squared Error:", mean_squared_error(y_test, y_pred))

    # Save the interest rate model if needed in the future
    joblib.dump(model, 'models/interest_rate_model.pkl')

    return model

# Main Execution
def main():
    df = load_data()
    df = feature_engineering(df)

    print("Training Cash Flow Risk Model...")
    train_cfrs_model(df)

    print("Training Interest Rate Model...")
    train_interest_rate_model(df)

    print("\n✅ Models saved successfully in the 'models' directory!")

if __name__ == "__main__":
    main()
