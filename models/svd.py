import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sentence_transformers import SentenceTransformer

class HybridRecommender:
    def __init__(self, k_factors=50):
        self.k = k_factors
        self.bert = SentenceTransformer('all-MiniLM-L6-v2')
        self.user_factors = None
        self.item_factors = None
        self.user_ids = None  # Will store string user IDs
        self.item_ids = None  # Will store string item IDs
        self.popular_items = None

    def _clean_id(self, id_value):
        """Convert ID to consistent string format"""
        if isinstance(id_value, float):
            return str(int(id_value)) if not np.isnan(id_value) else None
        return str(id_value).split('.')[0]  # Remove .0 if present

    def load_data(self, transactions_path, products_path):
        """Load and clean data with consistent ID formatting"""
        # Load transactions
        transactions = pd.read_csv(transactions_path)
        
        # Clean IDs and filter
        transactions['CustomerID'] = transactions['CustomerID'].apply(self._clean_id)
        transactions['StockCode'] = transactions['StockCode'].apply(self._clean_id)
        transactions = transactions.dropna(subset=['CustomerID', 'StockCode'])
        transactions = transactions[transactions['Quantity'] > 0]

        # Get unique IDs
        self.user_ids = np.unique(transactions['CustomerID'])
        self.item_ids = np.unique(transactions['StockCode'])

        # Create matrix
        user_map = {u: i for i, u in enumerate(self.user_ids)}
        item_map = {i: p for p, i in enumerate(self.item_ids)}
        
        self.matrix = np.zeros((len(user_map), len(item_map)))
        for _, row in transactions.iterrows():
            try:
                self.matrix[
                    user_map[row['CustomerID']],
                    item_map[row['StockCode']]
                ] += row['Quantity']
            except KeyError:
                continue

        # Store popular items
        self.popular_items = transactions['StockCode'].value_counts().index.tolist()

        # Load and clean products
        products = pd.read_csv(products_path)
        products['StockCode'] = products['StockCode'].apply(self._clean_id)
        products = products.dropna(subset=['StockCode', 'Description'])
        products['Description'] = products['Description'].astype(str)
        
        # Generate embeddings
        self.product_embeddings = {
            row['StockCode']: self.bert.encode(row['Description'])
            for _, row in products.iterrows()
        }

    def train(self):
        """Train SVD model"""
        U, sigma, Vt = svds(self.matrix, k=self.k)
        self.user_factors = U @ np.diag(sigma)
        self.item_factors = Vt.T

    def recommend(self, user_id, top_n=5):
        """Get recommendations with exact ID matching"""
        user_id = self._clean_id(user_id)
        if user_id is None:
            return [(item, 0.0) for item in self.popular_items[:top_n]]

        try:
            user_idx = np.where(self.user_ids == user_id)[0][0]
            scores = self.user_factors[user_idx] @ self.item_factors.T
            top_indices = np.argsort(scores)[-top_n:][::-1]
            return [(self.item_ids[i], scores[i]) for i in top_indices]
        except:
            return [(item, 0.0) for item in self.popular_items[:top_n]]

if __name__ == "__main__":
    print("Initializing recommender...")
    recommender = HybridRecommender()
    
    print("\nLoading data...")
    recommender.load_data(
        transactions_path="transactions_clean.csv",
        products_path="products_clean.csv"
    )
    
    print("\nTraining model...")
    recommender.train()
    
    print("\nTesting recommendations:")
    test_users = ["12347", "12346", 12347.0, "99999"]  # Test different formats
    for user in test_users:
        print(f"\nRecommendations for {user}:")
        print(recommender.recommend(user))