import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_processed_data(file_path="processed_data_for_xgboost.csv"):
    """Load processed data for XGBoost training."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading processed data: {str(e)}")
        raise

def train_xgboost_model(df):
    """Train an XGBoost model to predict customer abandonment."""
    try:
        # Define features and target
        features = ['TotalPurchases', 'AvgBasketSize', 'CancellationRate', 
                    'LastPurchaseDays', 'TotalSpent']
        target = 'Target'
        
        X = df[features]
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("\nXGBoost Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Save model
        with open("xgboost_model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        return model
    except Exception as e:
        print(f"Error training XGBoost model: {str(e)}")
        raise

def main():
    """Main function to train XGBoost model."""
    try:
        # Load processed data
        processed_data = load_processed_data()
        print(f"Loaded processed data shape: {processed_data.shape}")
        print("Columns:", processed_data.columns.tolist())
        print(f"Positive class ratio: {processed_data['Target'].mean() * 100:.2f}%")
        
        # Train model
        model = train_xgboost_model(processed_data)
        
        print("\nXGBoost training completed successfully!")
    except Exception as e:
        print(f"\nError in main: {str(e)}")

if __name__ == "__main__":
    main()