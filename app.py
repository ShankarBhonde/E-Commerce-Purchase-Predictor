import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ---- Train a simple model (only once) ----
data = pd.DataFrame({
    'time_spent': [30, 5, 12, 50, 3, 40, 8, 60],
    'pages_visited': [5, 1, 2, 7, 1, 6, 2, 9],
    'is_returning_user': [1, 0, 1, 1, 0, 1, 0, 1],
    'purchase': [1, 0, 0, 1, 0, 1, 0, 1]
})

# Create one extra feature
data['engagement_score'] = data['time_spent'] * data['pages_visited']

X = data.drop('purchase', axis=1)
y = data['purchase']

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# ---- Load model ----
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---- Streamlit App ----
st.title("E-Commerce Purchase Predictor")

time_spent = st.slider("Time Spent (minutes)", 0, 100, 10)
pages_visited = st.slider("Pages Visited", 1, 10, 2)
is_returning = st.radio("Returning User?", ["Yes", "No"])

is_returning_user = 1 if is_returning == "Yes" else 0
engagement_score = time_spent * pages_visited

# Prepare input
X_input = np.array([[time_spent, pages_visited, is_returning_user, engagement_score]])
X_input_scaled = scaler.transform(X_input)

if st.button("Predict"):
    prediction = model.predict(X_input_scaled)[0]
    prob = model.predict_proba(X_input_scaled)[0][1]

    if prediction == 1:
        st.success(f"Likely to Purchase (Confidence: {prob:.2f})")
    else:
        st.warning(f"Unlikely to Purchase (Confidence: {prob:.2f})")
