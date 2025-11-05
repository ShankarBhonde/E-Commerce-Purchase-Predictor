import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Train the model if not found
if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
    # Sample dataset
    data = pd.DataFrame({
        'time_spent': [30, 5, 12, 50, 3, 40, 8, 60],
        'pages_visited': [5, 1, 2, 7, 1, 6, 2, 9],
        'country': ['US', 'IN', 'US', 'UK', 'IN', 'US', 'UK', 'IN'],
        'is_returning_user': [1, 0, 1, 1, 0, 1, 0, 1],
        'purchase': [1, 0, 0, 1, 0, 1, 0, 1]
    })

    data = pd.get_dummies(data, columns=['country'], drop_first=True)
    data['engagement_score'] = data['time_spent'] * data['pages_visited']

    X = data.drop('purchase', axis=1)
    y = data['purchase']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🛒 E-Commerce Purchase Predictor")

# Input fields
time_spent = st.slider("Time Spent on Site (min)", 0, 100, 10)
pages_visited = st.slider("Pages Visited", 1, 10, 2)
country = st.selectbox("Country", ['US', 'IN', 'UK'])
is_returning = st.radio("Is Returning User?", ['Yes', 'No'])

# Encode inputs
country_IN = int(country == 'IN')
country_UK = int(country == 'UK')
is_returning_user = 1 if is_returning == 'Yes' else 0
engagement_score = time_spent * pages_visited

# Final input
X = np.array([[time_spent, pages_visited, is_returning_user,
               country_IN, country_UK, engagement_score]])
X_scaled = scaler.transform(X)

# Prediction
prediction = model.predict(X_scaled)[0]
prob = model.predict_proba(X_scaled)[0][1]

st.markdown("---")
if st.button("Predict"):
    if prediction == 1:
        st.success(f"🎯 Likely to Purchase (Confidence: {prob:.2f})")
    else:
        st.warning(f"🚫 Unlikely to Purchase (Confidence: {prob:.2f})")
