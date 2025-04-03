import numpy as np
import pandas as pd
from collections import defaultdict
import pickle

class LinUCB:
    def __init__(self, alpha=1.0):
        """
        Initialize with:
        - alpha: Exploration parameter
        - User features from XGBoost preprocessing
        """
        self.alpha = alpha
        self.A = {}  # Changed from defaultdict to regular dict
        self.b = {}   # Changed from defaultdict to regular dict
        self.discount_actions = [0, 5, 10]
        self._initialize_identity()

    def _initialize_identity(self):
        """Initialize identity matrices for new users"""
        self.identity_matrix = np.eye(5)
        self.zero_vector = np.zeros(5)

    def _get_user_arrays(self, user_id):
        """Safe getter for user-specific arrays"""
        if user_id not in self.A:
            self.A[user_id] = self.identity_matrix.copy()
            self.b[user_id] = self.zero_vector.copy()
        return self.A[user_id], self.b[user_id]

    def get_context(self, user_features):
        """Convert XGBoost features to context vector"""
        return np.array([
            user_features['TotalPurchases'] / 100,
            user_features['AvgBasketSize'] / 50,
            user_features['CancellationRate'],
            np.log(user_features['LastPurchaseDays'] + 1),
            user_features['AbandonmentRate']
        ])

    def recommend_discount(self, user_features):
        """Recommend discount for a user"""
        context = self.get_context(user_features)
        user_id = user_features['CustomerID']
        A_user, b_user = self._get_user_arrays(user_id)
        
        theta = np.linalg.inv(A_user) @ b_user
        uncertainty = self.alpha * np.sqrt(context @ np.linalg.inv(A_user) @ context)
        predicted_value = theta @ context
        
        if predicted_value + uncertainty < 0.3:
            return 0
        elif predicted_value + uncertainty < 0.7:
            return 5
        else:
            return 10

    def update_model(self, user_features, given_discount, purchased):
        """Update with observed outcome (purchased=1/0)"""
        context = self.get_context(user_features)
        user_id = user_features['CustomerID']
        A_user, b_user = self._get_user_arrays(user_id)
        
        self.A[user_id] = A_user + np.outer(context, context)
        self.b[user_id] = b_user + purchased * context

    def save_model(self, filename="linucb_model.pkl"):
        """Save model to pickle file"""
        with open(filename, "wb") as f:
            pickle.dump({
                'alpha': self.alpha,
                'A': self.A,
                'b': self.b,
                'discount_actions': self.discount_actions
            }, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load_model(cls, filename="linucb_model.pkl"):
        """Load model from pickle file"""
        with open(filename, "rb") as f:
            data = pickle.load(f)
        
        model = cls(data['alpha'])
        model.A = data['A']
        model.b = data['b']
        model.discount_actions = data['discount_actions']
        return model

if __name__ == "__main__":
    # Example usage
    user_features = pd.read_csv("user_features.csv")
    print(user_features.head())
    # Assuming user_features has a column 'CustomerID' and other relevant features
    user_data = user_features[user_features['CustomerID'] == 18282].iloc[0]
    
    discount_engine = LinUCB()
    discount = discount_engine.recommend_discount(user_data)
    print(f"Recommended discount: {discount}%")
    
    discount_engine.update_model(user_data, discount, purchased=1)
    discount_engine.save_model()