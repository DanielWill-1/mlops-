from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any
from models.svd_with_products import HybridRecommender
from models.linucb import LinUCB  # Ensure this import is correct
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models with error handling
def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading {model_path}: {str(e)}")

try:
    xgb_model = load_model('xgboost_model.pkl')
    recommender = load_model('hybrid_recommender.pkl')
    # Load LinUCB using its custom load_model method
    discount_engine = LinUCB.load_model('linucb_model.pkl')
    print("LinUCB model loaded successfully")
except Exception as e:
    print(f"Critical model loading error: {e}")
    exit(1)

# Mock product pricing database
PRODUCT_PRICES = {
    '85123A': 12.99,
    '71053': 8.50,
    '84029E': 5.99,
    '84406B': 7.25,
    '84029G': 6.99
}

def calculate_discount_price(product_code: str, discount_percent: float) -> float:
    base_price = PRODUCT_PRICES.get(product_code, 10.00)
    return round(base_price * (1 - discount_percent/100), 2)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        user_id = str(data.get('user_id', '12347'))
        user_features = data.get('features', {})
        
        required_features = ['TotalPurchases', 'AvgBasketSize', 'CancellationRate', 
                            'LastPurchaseDays', 'AbandonmentRate']
        if not all(k in user_features for k in required_features):
            missing = [k for k in required_features if k not in user_features]
            return jsonify({"error": f"Missing features: {missing}"}), 400

        try:
            user_features = {k: float(v) for k, v in user_features.items()}
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid feature value: {str(e)}"}), 400

        # 1. Predict abandonment probability
        xgb_input = np.array([[
            user_features['TotalPurchases'],
            user_features['AvgBasketSize'],
            user_features['CancellationRate'],
            user_features['LastPurchaseDays'],
            user_features['AbandonmentRate']
        ]])
        abandon_prob = float(xgb_model.predict_proba(xgb_input)[0][1])

        # 2. Generate recommendations
        recommendations = recommender.recommend(user_id)

        # 3. Calculate dynamic discount
        print(f"Calling recommend_discount with: {{'CustomerID': {user_id}, **{user_features}}}")
        discount = discount_engine.recommend_discount({'CustomerID': user_id, **user_features})
        print(f"Discount calculated: {discount}")

        # Prepare response
        response = {
            "user_id": user_id,
            "abandonment_probability": abandon_prob,
            "recommended_discount": f"{discount}%",
            "recommendations": [
                {
                    "product_code": item['code'],
                    "description": item['description'],
                    "base_score": round(float(item['score']), 2),
                    "original_price": PRODUCT_PRICES.get(item['code'], 10.00),
                    "discounted_price": calculate_discount_price(item['code'], discount)
                }
                for item in recommendations
            ]
        }

        return jsonify(response)

    except Exception as e:
        print("Error in prediction:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "xgboost": xgb_model is not None,
            "recommender": recommender is not None,
            "linucb": discount_engine is not None
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)