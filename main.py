import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime

###############################
# 🔹 Dynamic Interest Rate Logic
###############################
# Ideally, fetch the federal interest rate dynamically via an API.
# For now, we set a static example rate.
FEDERAL_INTEREST_RATE = 5.25  # Example: Adjust if needed

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
    df = pd.read_csv('data/sample1.csv')
    
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

        # Check if dataset is large enough to split
        if len(X) < 5:
            print("\n⚠️ Not enough data to split into train and test sets. Training on all data.")
            model = RandomForestClassifier(n_estimators=300, random_state=42)
            model.fit(X, y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Model evaluation
            y_pred = model.predict(X_test)
            print("\n✅ Model Performance:")
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
            print(classification_report(y_test, y_pred))

        # Save model
        joblib.dump(model, model_path)
        print(f"\n💾 Model saved to {model_path}")

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

    ###############################
    # 🔹 Updated Interest Rate Calculation
    ###############################
    # Adjusting interest rate margins based on risk category
    if risk_score > 75:
        risk_category = "High"
        interest_rate = FEDERAL_INTEREST_RATE + 6.75  # Higher risk, more margin
    elif risk_score > 40:
        risk_category = "Moderate"
        interest_rate = FEDERAL_INTEREST_RATE + 4.25
    else:
        risk_category = "Low"
        interest_rate = FEDERAL_INTEREST_RATE + 2.5  # Low risk, closer to base rate

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
