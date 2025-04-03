import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sentence_transformers import SentenceTransformer
import pickle

class HybridRecommender:
    def __init__(self, k_factors=50):
        self.k = k_factors
        self.bert = SentenceTransformer('all-MiniLM-L6-v2')
        self.user_factors = None
        self.item_factors = None
        self.user_ids = None
        self.item_ids = None
        self.product_embeddings = {}
        self.product_descriptions = {}  # New: Store descriptions
        self.popular_items = None

    def _clean_id(self, id_value):
        """Convert ID to consistent string format"""
        if isinstance(id_value, float):
            return str(int(id_value)) if not np.isnan(id_value) else None
        return str(id_value).split('.')[0]

    def load_data(self, transactions_path, products_path):
        """Load data with enhanced product info"""
        # Load transactions
        transactions = pd.read_csv(transactions_path)
        transactions['CustomerID'] = transactions['CustomerID'].apply(self._clean_id)
        transactions['StockCode'] = transactions['StockCode'].apply(self._clean_id)
        transactions = transactions.dropna(subset=['CustomerID', 'StockCode'])
        transactions = transactions[transactions['Quantity'] > 0]

        # Load products
        products = pd.read_csv(products_path)
        products['StockCode'] = products['StockCode'].apply(self._clean_id)
        products = products.dropna(subset=['StockCode', 'Description'])
        products['Description'] = products['Description'].astype(str)

        # Store descriptions
        self.product_descriptions = {
            row['StockCode']: row['Description']
            for _, row in products.iterrows()
        }

        # Generate embeddings
        self.product_embeddings = {
            code: self.bert.encode(desc)
            for code, desc in self.product_descriptions.items()
        }

        # Create matrix
        self.user_ids = np.unique(transactions['CustomerID'])
        self.item_ids = np.unique(transactions['StockCode'])
        
        user_map = {u: i for i, u in enumerate(self.user_ids)}
        item_map = {i: p for p, i in enumerate(self.item_ids)}
        
        self.matrix = np.zeros((len(user_map), len(item_map)))
        for _, row in transactions.iterrows():
            try:
                self.matrix[user_map[row['CustomerID']], item_map[row['StockCode']]] += row['Quantity']
            except KeyError:
                continue

        # Store popular items with descriptions
        self.popular_items = [
            (code, self.product_descriptions.get(code, "Unknown"))
            for code in transactions['StockCode'].value_counts().index.tolist()
        ]

    def train(self):
        """Train SVD model"""
        U, sigma, Vt = svds(self.matrix, k=self.k)
        self.user_factors = U @ np.diag(sigma)
        self.item_factors = Vt.T

    def recommend(self, user_id, top_n=5):
        """Get recommendations with full product info"""
        user_id = self._clean_id(user_id)
        if user_id is None:
            return [{
                'code': code,
                'description': desc,
                'score': 0.0,
                'type': 'popular'
            } for code, desc in self.popular_items[:top_n]]

        try:
            user_idx = np.where(self.user_ids == user_id)[0][0]
            scores = self.user_factors[user_idx] @ self.item_factors.T
            top_indices = np.argsort(scores)[-top_n:][::-1]
            
            return [{
                'code': self.item_ids[i],
                'description': self.product_descriptions.get(self.item_ids[i], "Unknown"),
                'score': float(scores[i]),
                'type': 'personalized'
            } for i in top_indices]
        except:
            return [{
                'code': code,
                'description': desc,
                'score': 0.0,
                'type': 'popular'
            } for code, desc in self.popular_items[:top_n]]

    def print_recommendations(self, recommendations):
        """Pretty-print recommendations"""
        print("\nRecommended Products:")
        print("-" * 50)
        for item in recommendations:
            print(f"Code: {item['code']}")
            print(f"Description: {item['description']}")
            print(f"Score: {item['score']:.2f} ({item['type']})")
            print("-" * 50)

    def train_and_save(self):
        self.train()
        with open("hybrid_recommender.pkl", "wb") as f:
            pickle.dump(self, f)
        print("Hybrid recommender saved to hybrid_recommender.pkl")

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
    test_users = ["12347", "12346", "99999"]
    for user in test_users:
        print(f"\nFor User {user}:")
        recs = recommender.recommend(user)
        recommender.print_recommendations(recs)
    recommender.train_and_save()