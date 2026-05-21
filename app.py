import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

st.title("Aircraft Engine RUL Prediction")

model_option = st.selectbox(
    "Choose Model",
    ["Machine Learning", "ANN", "LSTM"]
)

uploaded_file = st.file_uploader("Upload Test File")

if uploaded_file is not None:

    df = pd.read_csv(
        uploaded_file,
        sep=r"\s+",
        header=None,
        encoding="latin1",
        engine="python"
    )

    df = df.iloc[:, :26]

    scaler = joblib.load("models/scaler.pkl")

    X = df.drop([0, 1], axis=1)

    X_scaled = scaler.transform(X)

    if model_option == "Machine Learning":

        model = joblib.load("models/ml_model.pkl")

        prediction = model.predict(X_scaled)

    elif model_option == "ANN":

        model = load_model(
            "models/ann_model.h5",
            compile=False
        )

        prediction = model.predict(X_scaled)

    else:

        model = load_model(
            "models/lstm_model.h5",
            compile=False
        )

        X_scaled = np.reshape(
            X_scaled,
            (X_scaled.shape[0], 1, X_scaled.shape[1])
        )

        prediction = model.predict(X_scaled)

    st.subheader("Predicted RUL")

    st.dataframe(prediction[:20])