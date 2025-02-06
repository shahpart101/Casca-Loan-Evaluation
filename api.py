from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os
from io import StringIO
import subprocess
import openai
import re

# Initialize FastAPI
app = FastAPI()

# Configure CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Risk Model
model_path = 'backend/models/cash_flow_risk_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"\u2705 Model loaded successfully from: {model_path}")
else:
    raise FileNotFoundError("\u274C Model file not found. Ensure it's saved in the 'models' directory.")

#####################
# PDF Extraction Function
#####################
def extract_pdf_to_csv(pdf_file_path):
    output_txt = 'backend/temp_output.txt'
    output_csv = 'backend/temp_structured_output.csv'

    # Convert PDF to text
    subprocess.run(['pdftotext', pdf_file_path, output_txt], check=True)

    # Read extracted text
    with open(output_txt, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read()

    # Regular expressions for dates and amounts
    date_pattern = r'\d{2}-[A-Za-z]{3}-\d{4}'
    amount_pattern = r'\d{1,3}(?:,\d{3})*(?:\.\d{2})'

    # Split by date to identify transactions
    entries = re.split(f'({date_pattern})', text)
    transactions = []

    for i in range(1, len(entries) - 1, 2):
        date = entries[i].strip()
        details = entries[i + 1].strip().split('\n')

        if 'ACCOUNT' in details[0] or not details[0].strip():
            continue

        description = ' '.join([line.strip() for line in details if not re.match(date_pattern, line)])
        amounts = re.findall(amount_pattern, entries[i + 1])

        debit, credit, balance = '0', '0', '0'
        if len(amounts) == 2:
            debit, balance = amounts[0], amounts[1]
        elif len(amounts) == 3:
            debit, credit, balance = amounts[0], amounts[1], amounts[2]

        transactions.append([
            date, description, 
            debit.replace(',', ''), 
            credit.replace(',', ''), 
            balance.replace(',', '')
        ])

    # Write to CSV
    pd.DataFrame(transactions, columns=['Date', 'Description', 'Debit', 'Credit', 'Balance']).to_csv(output_csv, index=False)

    return output_csv

#####################
# /upload/ Endpoint
#####################
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_file_path = f"backend/temp_{file.filename}"
    with open(pdf_file_path, 'wb') as temp_pdf:
        temp_pdf.write(await file.read())

    try:
        csv_path = extract_pdf_to_csv(pdf_file_path)
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {e}")
    finally:
        os.remove(pdf_file_path)
        if os.path.exists(csv_path):
            os.remove(csv_path)

    # Data Cleaning
    df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0)
    df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0)
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce').fillna(0)
    df['Net Flow'] = df['Credit'] - df['Debit']

    # Feature Engineering
    features = pd.DataFrame({
        'AnomaliesCount': [len(df[(df['Debit'] > 5000) | (df['Credit'] > 10000)])],
        'AvgBalance': [df['Balance'].mean()],
        'AvgNetFlow': [df['Net Flow'].mean()],
        'TotalDeposits': [df['Credit'].sum()],
        'TotalWithdrawals': [df['Debit'].sum()],
        'FlowVolatility': [df['Net Flow'].std()]
    })

    try:
        risk_score = model.predict_proba(features)[:, 1][0] * 100
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    risk_category = 'High' if risk_score > 75 else 'Moderate' if risk_score > 40 else 'Low'
    interest_rate = 12.0 if risk_category == 'High' else 9.5 if risk_category == 'Moderate' else 7.0

    return JSONResponse(content={
        "RiskScore": round(risk_score, 2),
        "RiskCategory": risk_category,
        "InterestRate": interest_rate
    })

#####################
# /deep_analysis/ Endpoint
#####################
@app.post("/deep_analysis/")
async def deep_analysis(data: dict):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not found. Please set OPENAI_API_KEY.")

    prompt = f"""
    You are a financial analyst. Given these metrics:
    {data}

    Provide a detailed financial summary, red flags, and suggestions for improving financial health.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        text_result = response.choices[0].message['content'].strip()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content={"analysis": text_result})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)