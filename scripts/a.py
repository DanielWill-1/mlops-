import pandas as pd
import pickle
from datetime import timedelta
from sentence_transformers import SentenceTransformer

def load_data(file_paths=["Online Retail I.xlsx", "Online Retail II.xlsx"]):
    """Load and concatenate Excel files, clean column names."""
    try:
        dfs = [pd.read_excel(file) for file in file_paths]
        df = pd.concat(dfs, ignore_index=True)
        df.columns = df.columns.str.strip().str.replace(' ', '')
        df = df.loc[:, ~df.columns.duplicated()]
        
        required_cols = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 
                         'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def detect_abandonment(df):
    """
    Detect abandoned sessions/customers.
    Logic:
    - Cancelled: Invoice starts with 'C'.
    - Incomplete Session: Fewer than 2 positive-quantity items.
    - Inactive: No purchases >60 days from max date.
    """
    try:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df = df[df['CustomerID'].notna()].copy()
        
        df = df.sort_values(['CustomerID', 'InvoiceDate'])
        
        df['TimeDiff'] = df.groupby('CustomerID')['InvoiceDate'].diff().dt.total_seconds() / 60
        df['NewSession'] = (df['TimeDiff'] > 15) | df['TimeDiff'].isna()
        df['SessionID'] = df.groupby('CustomerID')['NewSession'].cumsum()
        
        df['IsCancelled'] = df['InvoiceNo'].astype(str).str.startswith('C')
        
        completed_sessions = df[(df['Quantity'] > 0) & ~df['IsCancelled']].groupby(
            ['CustomerID', 'SessionID']).agg(
                ItemCount=('StockCode', 'nunique'),
                TotalQuantity=('Quantity', 'sum')
        ).reset_index()
        completed_sessions['Completed'] = (completed_sessions['ItemCount'] >= 2) & (completed_sessions['TotalQuantity'] > 0)
        
        df = pd.merge(df, completed_sessions[['CustomerID', 'SessionID', 'Completed']], 
                      on=['CustomerID', 'SessionID'], how='left')
        df['Completed'] = df['Completed'].fillna(False)
        
        max_date = df['InvoiceDate'].max()
        last_purchase = df[(df['Quantity'] > 0) & ~df['IsCancelled']].groupby(
            'CustomerID')['InvoiceDate'].max().reset_index()
        last_purchase['DaysSinceLastPurchase'] = (max_date - last_purchase['InvoiceDate']).dt.days
        inactive_customers = last_purchase[last_purchase['DaysSinceLastPurchase'] > 60]['CustomerID']
        
        df['Abandoned'] = False
        df.loc[df['IsCancelled'], 'Abandoned'] = True
        df.loc[~df['Completed'], 'Abandoned'] = True
        df.loc[df['CustomerID'].isin(inactive_customers), 'Abandoned'] = True
        df.loc[df['Completed'] & ~df['CustomerID'].isin(inactive_customers), 'Abandoned'] = False
        
        print(f"Total rows: {len(df)}")
        print(f"Cancelled rows: {df['IsCancelled'].sum()}")
        print(f"Incomplete sessions rows: {(~df['Completed']).sum()}")
        print(f"Inactive customers: {len(inactive_customers)}")
        print(f"Abandoned rows: {df['Abandoned'].sum()}")
        print(f"Abandonment rate (raw): {df['Abandoned'].mean() * 100:.2f}%")
        
        return df
    except Exception as e:
        print(f"Error in abandonment detection: {str(e)}")
        raise

def generate_user_features(df):
    """Aggregate user-level features including abandonment rate."""
    try:
        user_features = df.groupby('CustomerID').agg(
            TotalPurchases=('InvoiceNo', 'nunique'),
            AvgBasketSize=('Quantity', lambda x: x.clip(0, 100).mean()),
            CancellationRate=('IsCancelled', 'mean'),
            LastPurchaseDays=('InvoiceDate', lambda x: (pd.Timestamp.now() - x.max()).days),
            TotalSpent=('Quantity', lambda x: (x * df.loc[x.index, 'UnitPrice']).sum()),
            AbandonmentRate=('Abandoned', 'mean')
        ).reset_index()
        return user_features
    except Exception as e:
        print(f"Error generating user features: {str(e)}")
        raise

def generate_product_embeddings(df):
    """Create BERT embeddings for unique product descriptions."""
    try:
        product_descriptions = df[['StockCode', 'Description']].drop_duplicates()
        product_descriptions['Description'] = product_descriptions['Description'].fillna("Unknown")
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(product_descriptions['Description'].tolist(), 
                                 show_progress_bar=True)
        product_embeddings = dict(zip(product_descriptions['StockCode'], embeddings))
        return product_embeddings
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        raise

def preprocess_data():
    """Execute preprocessing and save processed data."""
    try:
        # Load data
        df = load_data()
        print(f"Raw data shape: {df.shape}")
        print("Available columns:", df.columns.tolist())
        
        # Detect abandonment
        df = detect_abandonment(df)
        abandonment_rate = df['Abandoned'].mean() * 100
        print(f"\nVerified abandonment rate: {abandonment_rate:.2f}%")
        
        # Generate user features
        user_features = generate_user_features(df)
        print("\nSample user features with abandonments:")
        print(user_features[user_features['AbandonmentRate'] > 0].head(3))
        
        # Prepare processed data for XGBoost
        processed_data = user_features.copy()
        processed_data['Target'] = (processed_data['AbandonmentRate'] > 0.5).astype(int)  # Binary target
        features = ['TotalPurchases', 'AvgBasketSize', 'CancellationRate', 
                    'LastPurchaseDays', 'TotalSpent', 'Target']
        processed_data = processed_data[features]  # Only keep features and target
        
        # Generate product embeddings
        print("\nGenerating BERT embeddings...")
        product_embeddings = generate_product_embeddings(df)
        
        # Save outputs
        user_features.to_csv("user_features.csv", index=False)
        processed_data.to_csv("processed_data_for_xgboost.csv", index=False)
        with open("product_embeddings.pkl", "wb") as f:
            pickle.dump(product_embeddings, f)
        
        print("\nProcessing completed successfully!")
        return True
    except Exception as e:
        print(f"\nError during preprocessing: {str(e)}")
        return False

if __name__ == "__main__":
    preprocess_data()