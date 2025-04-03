from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load all models
with open('xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
    
with open('hybrid_recommender.pkl', 'rb') as f:
    recommender = pickle.load(f)

with open('linucb_model.pkl', 'rb') as f:
    discount_engine = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user data from request
    data = request.json
    user_id = str(data.get('user_id', '12347'))
    user_features = data.get('features', {})
    
    # 1. Predict abandonment probability
    abandon_prob = xgb_model.predict_proba([list(user_features.values())])[0][1]
    
    # 2. Generate recommendations
    recommendations = recommender.recommend(user_id)
    
    # 3. Calculate dynamic discount
    discount = discount_engine.recommend_discount(user_features)
    
    return jsonify({
        "user_id": user_id,
        "abandonment_probability": float(abandon_prob),
        "recommended_discount": f"{discount}%",
        "recommendations": [
            {
                "product_code": item['code'],
                "description": item['description'],
                "base_score": round(item['score'], 2),
                "discounted_price": calculate_discount_price(item['code'], discount)
            }
            for item in recommendations
        ]
    })

def calculate_discount_price(product_code, discount_percent):
    # Mock pricing function - replace with your actual pricing logic
    base_prices = {'85123A': 12.99, '71053': 8.50, '84029E': 5.99}
    price = base_prices.get(product_code, 10.00)
    return round(price * (1 - discount_percent/100), 2)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)