# ml project E commerce optimization using ml strategies 

E-Commerce Cart Optimization
An intelligent system to enhance online shopping experiences by reducing cart abandonment and boosting conversion rates through dynamic discounts, predictive analytics, and personalized product recommendations.

Overview
This project addresses the issue of shopping cart abandonment using machine learning and reinforcement learning techniques. It predicts when a customer is likely to leave their cart and responds dynamically by offering targeted discounts and personalized product suggestions.

Key Features
Cart Abandonment Prediction using XGBoost and TFT

Dynamic Discounting with LinUCB (Contextual Bandit) + PPO

Personalized Recommendations using a Hybrid Collaborative Filtering + BERT Embeddings model ( SVD + BERT)

Customer behavior insights and purchase pattern analysis ( retargeting )

Tech Stack

Python, NumPy, Pandas, scikit learn

XGBoost for classification

LinUCB for adaptive discounts

Sentence-BERT and SVD for hybrid recommendations

Scikit-learn, Matplotlib (optional for visualization)

Goals
Reduce cart abandonment rate

Increase order completion

Deliver personalized, adaptive shopping experiences

Datasets Used
UCI Online Retail I & II

Retail Rocket Dataset

Olist E-Commerce Dataset

E-Commerce Behavior Data

How to Use
Clone the repository

Run the data preprocessing and training scripts

Launch the recommendation and discount modules

View personalized output for test users

Results: so far
Histroical discount recommendations 

Accurate product suggestions based on user intent and history

Future Improvements TODO!!!
Integrate real-time tracking from live websites

Expand discount actions using reinforcement learning 

Add user-facing dashboard and API for business integration
 
Add real time clickstream data

Add proper UI dashboards to use the models 