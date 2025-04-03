import pandas as pd

def clean_transaction_data(input_path, output_path):
    # Load data with proper data types
    df = pd.read_csv(input_path, dtype={'CustomerID': str, 'StockCode': str})
    
    # 1. Remove missing CustomerIDs (24.9% of data)
    clean_df = df.dropna(subset=['CustomerID'])
    
    # 2. Remove negative/zero quantities
    clean_df = clean_df[clean_df['Quantity'] > 0]
    
    # 3. Remove refunds/voids (invoices starting with 'C'
    
    # 4. Remove non-product codes (e.g., discounts, shipping)
    clean_df = clean_df[clean_df['StockCode'].str.match('^[a-zA-Z0-9]{5,6}$')]
    
    # Save cleaned data
    clean_df.to_csv(output_path, index=False)
    
    print(f"Cleaning report:")
    print(f"- Original rows: {len(df):,}")
    print(f"- Removed missing CustomerIDs: {len(df) - len(clean_df):,}")
    print(f"- Final clean rows: {len(clean_df):,}")

clean_transaction_data("transactions.csv", "transactions_clean.csv")