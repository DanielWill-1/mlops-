import pickle

with open('linucb_model.pkl', 'rb') as f:
    content = pickle.load(f)
    print(type(content))
    print(content)