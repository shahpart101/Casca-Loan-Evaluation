import os
import re
import csv
import subprocess

# Specify input and output files
input_pdf = 'sample 1.pdf'
output_txt = 'output.txt'
output_csv = 'structured_output.csv'

# Step 1: Convert PDF to text if not already done
if not os.path.exists(output_txt):
    try:
        print("🔄 Extracting text from PDF...")
        subprocess.run(['pdftotext', input_pdf, output_txt], check=True)
        print("✅ Text extraction complete!")
    except subprocess.CalledProcessError:
        print(f"❌ Error: Failed to extract text from {input_pdf}. Ensure 'pdftotext' is installed.")
        exit(1)

# Step 2: Read extracted text
try:
    with open(output_txt, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read()
except FileNotFoundError:
    print(f"❌ Error: '{output_txt}' not found.")
    exit(1)

# Step 3: Regular expressions for dates and amounts
date_pattern = r'\d{2}-[A-Za-z]{3}-\d{4}'
amount_pattern = r'\d{1,3}(?:,\d{3})*(?:\.\d{2})'

# Step 4: Split by date to identify transactions
entries = re.split(f'({date_pattern})', text)
transactions = []

for i in range(1, len(entries) - 1, 2):
    date = entries[i].strip()
    details = entries[i + 1].strip().split('\n')
    
    # Skip non-transactional lines
    if 'ACCOUNT' in details[0] or 'IMPORTANT MESSAGE' in details[0] or not details[0].strip():
        continue

    # Clean description by joining relevant lines
    description = ' '.join([line.strip() for line in details if not re.match(date_pattern, line) and not re.match(amount_pattern, line)])
    
    # Find amounts (debits, credits, balances)
    amounts = re.findall(amount_pattern, entries[i + 1])

    # Identify Debit, Credit, and Balance
    debit, credit, balance = '', '', ''
    if len(amounts) == 2:
        debit, balance = amounts[0], amounts[1]
    elif len(amounts) == 3:
        debit, credit, balance = amounts[0], amounts[1], amounts[2]
    else:
        debit = credit = balance = 'N/A'

    transactions.append([date, description, debit, credit, balance])

# Step 5: Write results to CSV
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Date', 'Description', 'Debit', 'Credit', 'Balance'])
    writer.writerows(transactions)

print(f"✅ Transactions successfully extracted to '{output_csv}'")
