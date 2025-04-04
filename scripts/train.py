import pickle
import numpy as np

with open('xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

xgb_input = np.array([[40, 24.5, 0.1, 14, 1]])  # Example input
prob = xgb_model.predict_proba(xgb_input)[0][1]
print(f"XGBoost probability: {prob}")