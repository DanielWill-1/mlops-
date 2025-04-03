import pandas as pd
# Load the cleaned data

products = pd.read_csv("products.csv")
products['Description'] = products['Description'].fillna('Unknown').astype(str)
products.to_csv("products_clean.csv", index=False)