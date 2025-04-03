import subprocess

# Train and save all models sequentially
print("Training XGBoost...")
subprocess.run(["python", "xgboost_model.py"])

print("\nTraining Hybrid Recommender...")
subprocess.run(["python", "hybrid_recommender.py"])

print("\nTraining LinUCB...") 
subprocess.run(["python", "linucb.py"])

print("\nAll models trained and saved:")
print("- xgboost_model.pkl")
print("- hybrid_recommender.pkl") 
print("- linucb_model.pkl")