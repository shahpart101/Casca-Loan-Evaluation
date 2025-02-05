import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime

def calculate_cashflow_features(df):
    """Calculate enhanced financial metrics from transaction data"""
    features = {
        # Original features
        'AvgBalance': df['Balance'].mean() if 'Balance' in df.columns else 0,
        'AvgNetFlow': (df['Credit'].sum() - df['Debit'].sum()) / len(df) if len(df) > 0 else 0,
        'TotalDeposits': df['Credit'].sum() if 'Credit' in df.columns else 0,
        'TotalWithdrawals': df['Debit'].sum() if 'Debit' in df.columns else 0,
        'FlowVolatility': df['Net Flow'].std() if 'Net Flow' in df.columns else 0,
        'AnomaliesCount': len(df[(df['Debit'] > 5000) | (df['Credit'] > 10000)]),
        
        # New features
        'BalanceVolatility': df['Balance'].std() if 'Balance' in df.columns else 0,
        'TransactionCount': len(df),
        'AvgTransactionSize': df['Credit'].mean() if 'Credit' in df.columns else 0,
        'LargeTransactionsRatio': len(df[df['Credit'] > df['Credit'].mean() * 2]) / len(df) if len(df) > 0 else 0
    }
    return pd.DataFrame([features])

def load_and_process_files(file_paths):
    """Load and process multiple CSV files"""
    all_data = []
    
    for file_path, default_status in file_paths:
        try:
            df = pd.read_csv(file_path)
            features = calculate_cashflow_features(df)
            features['Default'] = default_status
            features['BorrowerID'] = df['BorrowerID'].iloc[0] if 'BorrowerID' in df.columns else len(all_data) + 1
            all_data.append(features)
            print(f"\n✅ Successfully processed {file_path}")
        except Exception as e:
            print(f"\n⚠️ Error processing {file_path}: {e}")
            continue
    
    return pd.concat(all_data, ignore_index=True) if all_data else None

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    # Define training files and their default status (0 = good, 1 = default)
    training_files = [
        ('sample1.csv', 0),  # Good credit history
        ('sample2.csv', 1),  # Default history
        ('sample3.csv', 0)   # Another good credit history
    ]

    # Load and process all training files
    print("\n🔄 Processing training files...")
    training_data = load_and_process_files(training_files)
    
    if training_data is None:
        print("\n❌ No valid training data found. Please check your input files.")
        return

    # Define all features we want to use
    required_features = [
        'AnomaliesCount',
        'AvgBalance',
        'AvgNetFlow',
        'TotalDeposits',
        'TotalWithdrawals',
        'FlowVolatility',
        'BalanceVolatility',
        'TransactionCount',
        'AvgTransactionSize',
        'LargeTransactionsRatio'
    ]

    # Add missing features with zeros if necessary
    for feat in required_features:
        if feat not in training_data.columns:
            training_data[feat] = 0

    # Split features and target
    X = training_data[required_features]
    y = training_data['Default']

    # Model Training
    model_path = 'models/cash_flow_risk_model.pkl'
    print("\n🚀 Training new CFRS model...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(
        n_estimators=200,  # More trees
        max_depth=10,      # Control overfitting
        min_samples_split=5,
        class_weight='balanced',  # Handle imbalanced classes
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, model_path)
    
    # Model evaluation
    y_pred = model.predict(X_test)
    print("\n✅ Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': required_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\n🎯 Feature Importance:")
    print(feature_importance)

    # Process new application
    print("\n🔍 Processing new application...")
    new_application = pd.read_csv('new_application.csv')
    features = calculate_cashflow_features(new_application)
    
    # Ensure features match training features
    for feat in required_features:
        if feat not in features.columns:
            features[feat] = 0
    features = features[required_features]

    # Generate predictions
    try:
        risk_score = model.predict_proba(features)[:, 1][0] * 100
        risk_category = 'High' if risk_score > 75 else 'Moderate' if risk_score > 40 else 'Low'
        interest_rate = 12.0 if risk_category == 'High' else 9.5 if risk_category == 'Moderate' else 7.0

        results = pd.DataFrame({
            'BorrowerID': [new_application['BorrowerID'].iloc[0] if 'BorrowerID' in new_application.columns else 1],
            'RiskScore': [round(risk_score, 2)],
            'RiskCategory': [risk_category],
            'InterestRate': [interest_rate],
            'EvaluationDate': [datetime.today().strftime('%Y-%m-%d')]
        })
        
        results.to_csv('output/loan_evaluation_scores.csv', index=False)
        print("\n📊 Evaluation Results:")
        print(results.to_string(index=False))

    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")

if __name__ == "__main__":
    main()