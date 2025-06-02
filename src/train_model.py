import pandas as pd
import joblib
import os
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import streamlit as st

def train_and_save_model():
    # Load training data
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv")

    # Split into training and validation sets
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Scale inputs and targets
    input_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_tr_scaled = input_scaler.fit_transform(X_tr)
    X_val_scaled = input_scaler.transform(X_val)

    y_tr_scaled = target_scaler.fit_transform(y_tr.values.reshape(-1, 1))
    y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1))

    # Train MLP model
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                       max_iter=500, random_state=42)
    mlp.fit(X_tr_scaled, y_tr_scaled.ravel())

    # Evaluate on validation set
    y_pred_scaled = mlp.predict(X_val_scaled)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    print(f"Validation MAE (residual): {mae:.2f}")
    print(f"Validation RMSE (residual): {rmse:.2f}")

    # Save model and scalers
    os.makedirs("models", exist_ok=True)
    joblib.dump(mlp, "models/mlp_model.pkl")
    joblib.dump(input_scaler, "models/input_scaler.pkl")
    joblib.dump(target_scaler, "models/target_scaler.pkl")
