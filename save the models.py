import pickle
import numpy as np
from datetime import datetime
import pandas as pd
from models import train_xgboost
from models import svd_with_products  # Your Hybrid Recommender implementation
from models import linucb  # Your LinUCB implementation

def train_and_save_models():
    # 1. Load your preprocessed data
    user_features = pd.read_csv("user_features.csv")
    transactions = pd.read_csv("transactions_clean.csv")
    products = pd.read_csv("products_clean.csv")

    # 2. Train XGBoost Abandonment Model
    print("Training XGBoost model...")
    xgb_model = train_xgboost()
    X = user_features.drop(['CustomerID', 'AbandonmentRate'], axis=1)
    y = user_features['AbandonmentRate']
    xgb_model.fit(X, y)
    
    # 3. Train Hybrid Recommender
    print("Training Hybrid Recommender...")
    recommender = svd_with_products()
    recommender.load_data("data/transactions_clean.csv", "data/products_clean.csv")
    recommender.train()
    
    # 4. Train LinUCB Discount Engine
    print("Training LinUCB model...")
    discount_engine = linucb()
    
    # Simulate some training (replace with your actual training data)
    for _, row in user_features.iterrows():
        discount_engine.update_model(
            user_data=row.to_dict(),
            given_discount=np.random.choice([0, 5, 10]),
            purchased=np.random.choice([0, 1])
        )
    
    # 5. Save all models with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nSaving models...")
    with open(f'models/xgboost_{timestamp}.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
        
    with open(f'models/recommender_{timestamp}.pkl', 'wb') as f:
        pickle.dump(recommender, f)
        
    with open(f'models/linucb_{timestamp}.pkl', 'wb') as f:
        pickle.dump(discount_engine, f)
    
    # Create symlink to latest
    import os
    for model in ['xgboost', 'recommender', 'linucb']:
        src = f'models/{model}_{timestamp}.pkl'
        dst = f'models/{model}_latest.pkl'
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    
    print(f"Models saved to models/ with timestamp: {timestamp}")

if __name__ == "__main__":
    train_and_save_models()