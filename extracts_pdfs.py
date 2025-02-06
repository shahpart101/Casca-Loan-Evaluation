import os
import re
import csv
import subprocess

def extract_pdf_to_csv(input_pdf, output_txt="output.txt", output_csv="structured_output.csv"):
    """
    Extracts transaction data from a PDF file and writes it to a CSV.

    Parameters:
        input_pdf (str): Path to the input PDF file.
        output_txt (str): Path to the intermediate text file.
        output_csv (str): Path where the CSV output will be saved.

    Returns:
        str: The path to the CSV file with extracted transaction data.
    """
    # Step 1: Convert PDF to text (if not already done)
    if not os.path.exists(output_txt):
        try:
            print("🔄 Extracting text from PDF...")
            subprocess.run(['pdftotext', input_pdf, output_txt], check=True)
            print("✅ Text extraction complete!")
        except subprocess.CalledProcessError:
            raise RuntimeError(f"❌ Error: Failed to extract text from {input_pdf}. Ensure 'pdftotext' is installed.")

    # Step 2: Read the extracted text
    try:
        with open(output_txt, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Error: '{output_txt}' not found.")

    # Step 3: Define regular expressions for dates and amounts
    date_pattern = r'\d{2}-[A-Za-z]{3}-\d{4}'
    amount_pattern = r'\d{1,3}(?:,\d{3})*(?:\.\d{2})'

    # Step 4: Split the text by date to identify transactions
    entries = re.split(f'({date_pattern})', text)
    transactions = []
    for i in range(1, len(entries) - 1, 2):
        date = entries[i].strip()
        details = entries[i + 1].strip().split('\n')

        # Skip non-transactional lines
        if 'ACCOUNT' in details[0] or 'IMPORTANT MESSAGE' in details[0] or not details[0].strip():
            continue

        # Build a description by joining lines that do not match a date or amount pattern
        description = ' '.join([line.strip() for line in details 
                                 if not re.match(date_pattern, line) and not re.match(amount_pattern, line)])
        
        # Find all amounts in the transaction block
        amounts = re.findall(amount_pattern, entries[i + 1])
        
        # Identify Debit, Credit, and Balance based on number of amounts found
        debit, credit, balance = '', '', ''
        if len(amounts) == 2:
            debit, balance = amounts[0], amounts[1]
        elif len(amounts) == 3:
            debit, credit, balance = amounts[0], amounts[1], amounts[2]
        else:
            debit = credit = balance = 'N/A'
        
        transactions.append([date, description, debit.replace(',', ''), credit.replace(',', ''), balance.replace(',', '')])

    # Step 5: Write the transactions to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Date', 'Description', 'Debit', 'Credit', 'Balance'])
        writer.writerows(transactions)

    print(f"✅ Transactions successfully extracted to '{output_csv}'")
    return output_csv

if __name__ == "__main__":
    input_pdf = 'sample 1.pdf'  # Change to your actual PDF filename if necessary
    extract_pdf_to_csv(input_pdf)
