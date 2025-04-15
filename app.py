
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

st.title("🌧️ Rainfall Prediction using Machine Learning")

st.markdown("""
Upload your rainfall dataset in CSV format. The model will be trained and used to predict rainfall.
""")

uploaded_file = st.file_uploader("Upload your Rainfall.csv file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("📊 Data Preview", data.head())

    # Clean column names
    data.columns = data.columns.str.strip()

    # Drop unwanted column if present
    if "day" in data.columns:
        data = data.drop(columns=["day"])

    # Handle missing values
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype == "object":
                data[col] = data[col].fillna(data[col].mode()[0])
            else:
                data[col] = data[col].fillna(data[col].median())

    st.write("✅ Cleaned Data", data.head())

    # Define features and label
    if "RainTomorrow" not in data.columns:
        st.error("🛑 'RainTomorrow' column (label) not found in dataset.")
    else:
        X = data.drop("RainTomorrow", axis=1)
        y = data["RainTomorrow"]

        # Encode categorical variables
        X = pd.get_dummies(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model trained with accuracy: {acc:.2f}")

        st.subheader("📈 Make a Prediction")
        input_data = {}
        for col in X.columns:
            if "int" in str(X[col].dtype) or "float" in str(X[col].dtype):
                input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))
            else:
                input_data[col] = st.selectbox(f"{col}", options=X[col].unique())

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            # Align with training columns
            input_df = input_df.reindex(columns=X.columns, fill_value=0)
            prediction = model.predict(input_df)[0]
            st.success(f"🌂 Predicted Rain Tomorrow: {prediction}")
