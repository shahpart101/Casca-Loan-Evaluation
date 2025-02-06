from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os
import subprocess
import re
from openai import OpenAI

# Initialize FastAPI
app = FastAPI()

# ✅ Root endpoint to confirm backend is live
@app.get("/")
def home():
    return {"message": "Casca Backend is Running!"}

# ✅ Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load Risk Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "backend/models/cash_flow_risk_model.pkl")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
else:
    raise FileNotFoundError("❌ Model file not found. Ensure it's in the 'models' directory.")

# ✅ PDF Extraction Function
def extract_pdf_to_csv(pdf_file_path):
    output_txt = "backend/temp_output.txt"
    output_csv = "backend/temp_structured_output.csv"

    # Check if pdftotext is installed
    if subprocess.run(["which", "pdftotext"], capture_output=True, text=True).returncode != 0:
        raise FileNotFoundError("❌ `pdftotext` is not installed. Install via `brew install poppler` (Mac) or `apt install poppler-utils` (Linux).")

    # Convert PDF to text
    try:
        subprocess.run(['pdftotext', pdf_file_path, output_txt], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"❌ PDF extraction failed: {e}")

    # Read extracted text
    with open(output_txt, "r", encoding="utf-8", errors="ignore") as file:
        text = file.read()

    # Extract data using regex
    date_pattern = r'\d{2}-[A-Za-z]{3}-\d{4}'
    amount_pattern = r'\d{1,3}(?:,\d{3})*(?:\.\d{2})'
    entries = re.split(f'({date_pattern})', text)
    transactions = []

    for i in range(1, len(entries) - 1, 2):
        date = entries[i].strip()
        details = entries[i + 1].strip().split("\n")
        description = " ".join([line.strip() for line in details if not re.match(date_pattern, line)])
        amounts = re.findall(amount_pattern, entries[i + 1])

        debit, credit, balance = "0", "0", "0"
        if len(amounts) == 2:
            debit, balance = amounts[0], amounts[1]
        elif len(amounts) == 3:
            debit, credit, balance = amounts[0], amounts[1], amounts[2]

        transactions.append([
            date, description, 
            debit.replace(",", ""), 
            credit.replace(",", ""), 
            balance.replace(",", "")
        ])

    # ✅ Write to CSV
    pd.DataFrame(transactions, columns=["Date", "Description", "Debit", "Credit", "Balance"]).to_csv(output_csv, index=False)
    return output_csv

# ✅ /upload/ Endpoint
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    print(f"📂 Received file: {file.filename}")  # Debugging print

    # Check if it's a PDF
    if not file.filename.endswith('.pdf'):
        print(f"❌ File rejected: {file.filename}")  # Debugging print
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_file_path = f"temp_{file.filename}"
    with open(pdf_file_path, 'wb') as temp_pdf:
        temp_pdf.write(await file.read())

    print(f"✅ Saved PDF file at: {pdf_file_path}")

    try:
        csv_path = extract_pdf_to_csv(pdf_file_path)
        print(f"✅ Extracted CSV: {csv_path}")
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ ERROR during PDF parsing: {e}")  # Debugging
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {e}")
    finally:
        os.remove(pdf_file_path)
        if os.path.exists(csv_path):
            os.remove(csv_path)

    # ✅ Data Cleaning
    df["Debit"] = pd.to_numeric(df["Debit"], errors="coerce").fillna(0)
    df["Credit"] = pd.to_numeric(df["Credit"], errors="coerce").fillna(0)
    df["Balance"] = pd.to_numeric(df["Balance"], errors="coerce").fillna(0)
    df["Net Flow"] = df["Credit"] - df["Debit"]

    # ✅ Feature Engineering
    features = pd.DataFrame({
        "AnomaliesCount": [len(df[(df["Debit"] > 5000) | (df["Credit"] > 10000)])],
        "AvgBalance": [df["Balance"].mean()],
        "AvgNetFlow": [df["Net Flow"].mean()],
        "TotalDeposits": [df["Credit"].sum()],
        "TotalWithdrawals": [df["Debit"].sum()],
        "FlowVolatility": [df["Net Flow"].std()]
    }).fillna(0)

    # ✅ Predict Risk Score
    try:
        risk_score = model.predict_proba(features)[:, 1][0] * 100
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    risk_category = "High" if risk_score > 75 else "Moderate" if risk_score > 40 else "Low"
    interest_rate = 12.0 if risk_category == "High" else 9.5 if risk_category == "Moderate" else 7.0

    return JSONResponse(content={
        "RiskScore": round(risk_score, 2),
        "RiskCategory": risk_category,
        "InterestRate": interest_rate
    })

# ✅ /deep_analysis/ (Fixed OpenAI Integration)
@app.post("/deep_analysis/")
async def deep_analysis(data: dict):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not client.api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not found. Please set OPENAI_API_KEY.")

    prompt = f"""
You are a financial analyst for a fintech company. Given the following financial metrics:
{data}

1. Assess the borrower's financial stability.
2. Determine if Casca should approve or decline a loan.
3. Recommend a risk-adjusted interest rate based on market conditions.
4. Explain how Casca can maximize profits while minimizing risk.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        text_result = response.choices[0].message.content.strip()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

    return JSONResponse(content={"analysis": text_result})

# ✅ Start Uvicorn Server for Local Testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
