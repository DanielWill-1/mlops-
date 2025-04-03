import pandas as pd

def check_data_quality(file_path):
    df = pd.read_csv(file_path)
    
    print("\nData Quality Report:")
    print(f"Total rows: {len(df)}")
    print(f"Missing CustomerID: {df['CustomerID'].isna().sum()}")
    print(f"Missing StockCode: {df['StockCode'].isna().sum()}")
    print(f"Invalid Quantities: {(df['Quantity'] <= 0).sum()}")
    
    # Show problematic rows
    print("\nSample problematic rows:")
    print(df[df['CustomerID'].isna() | df['StockCode'].isna() | (df['Quantity'] <= 0)].head())

def validate_clean_data(file_path):
    df = pd.read_csv(file_path)
    print("\nData Validation Report:")
    assert df['CustomerID'].isna().sum() == 0, "CustomerID missing!"
    assert (df['Quantity'] <= 0).sum() == 0, "Invalid quantities!"
    assert df['StockCode'].isna().sum() == 0, "StockCode missing!"
    print("✅ Data validation passed!")

if __name__ == "__main__":
    check_data_quality("transactions_clean.csv")
    validate_clean_data("transactions_clean.csv")
    # Load the cleaned data and validate it