import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime

def calculate_cashflow_features(df):
    """Calculate financial metrics from transaction data"""
    features = {
        'AvgBalance': df['Balance'].mean() if 'Balance' in df.columns else 0,
        'AvgNetFlow': (df['Credit'].sum() - df['Debit'].sum()) / len(df) if len(df) > 0 else 0,
        'TotalDeposits': df['Credit'].sum() if 'Credit' in df.columns else 0,
        'TotalWithdrawals': df['Debit'].sum() if 'Debit' in df.columns else 0,
        'FlowVolatility': df['Net Flow'].std() if 'Net Flow' in df.columns else 0,
        'AnomaliesCount': len(df[(df['Debit'] > 5000) | (df['Credit'] > 10000)])
    }
    return pd.DataFrame([features])

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    # Load and preprocess data
    df = pd.read_csv('sample1.csv')
    
    # Handle missing columns
    if 'Default' not in df.columns:
        print("\n⚠️ Creating placeholder 'Default' column")
        df['Default'] = 0
        
    if 'BorrowerID' not in df.columns:
        print("\n⚠️ Creating placeholder 'BorrowerID'")
        df['BorrowerID'] = 1

    # Feature Engineering
    features = calculate_cashflow_features(df)
    
    # Ensure consistent feature order
    required_features = [
        'AnomaliesCount',
        'AvgBalance',
        'AvgNetFlow',
        'TotalDeposits',
        'TotalWithdrawals',
        'FlowVolatility'
    ]

    # Add missing features with zeros if necessary
    for feat in required_features:
        if feat not in features.columns:
            features[feat] = 0

    features = features[required_features]

    # Model Training/Loading
    model_path = 'models/cash_flow_risk_model.pkl'
    if not os.path.exists(model_path):
        print("\n🚀 Training new CFRS model...")
        
        # Generate synthetic training data with correct feature structure
        synthetic_data = pd.DataFrame({
            'AnomaliesCount': np.random.randint(0, 5, 1000),
            'AvgBalance': np.random.uniform(1000, 50000, 1000),
            'AvgNetFlow': np.random.uniform(-5000, 10000, 1000),
            'TotalDeposits': np.random.uniform(1000, 50000, 1000),
            'TotalWithdrawals': np.random.uniform(500, 45000, 1000),
            'FlowVolatility': np.random.uniform(50, 5000, 1000),
            'Default': np.random.randint(0, 2, 1000)
        })

        X = synthetic_data[required_features]
        y = synthetic_data['Default']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        joblib.dump(model, model_path)
        
        # Model evaluation
        y_pred = model.predict(X_test)
        print("\n✅ Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(classification_report(y_test, y_pred))
    else:
        print("\n🔍 Loading existing CFRS model...")
        model = joblib.load(model_path)

    # Generate predictions
    try:
        risk_score = model.predict_proba(features)[:, 1][0] * 100
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        print("Current features:", features.columns.tolist())
        print("Feature order should be:", required_features)
        # Delete the model file to force retraining
        os.remove(model_path)
        print("\n🔄 Model file deleted. Please run the script again to retrain the model.")
        return

    # Determine loan terms
    risk_category = 'High' if risk_score > 75 else 'Moderate' if risk_score > 40 else 'Low'
    interest_rate = 12.0 if risk_category == 'High' else 9.5 if risk_category == 'Moderate' else 7.0

    # Save results
    results = pd.DataFrame({
        'BorrowerID': [df['BorrowerID'].iloc[0]],
        'RiskScore': [round(risk_score, 2)],
        'RiskCategory': [risk_category],
        'InterestRate': [interest_rate],
        'EvaluationDate': [datetime.today().strftime('%Y-%m-%d')]
    })
    
    results.to_csv('output/loan_evaluation_scores.csv', index=False)
    print("\n📊 Evaluation Results:")
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()