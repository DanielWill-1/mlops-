import subprocess

# Train and save all models sequentially
print("Training XGBoost...")
subprocess.run(["python", "models\train_xgboost.py"])

print("\nTraining Hybrid Recommender...")
subprocess.run(["python", "models\svd_with_products.py"])

print("\nTraining LinUCB...") 
subprocess.run(["python", "models\linucb.py"])

print("\nAll models trained and saved:")
print("- xgboost_model.pkl")
print("- hybrid_recommender.pkl") 
print("- linucb_model.pkl")