 with open("xgboost_model.pkl", "wb") as f:
            pickle.dump(model, f)