import numpy as np
import pandas as pd
import joblib
from src.observed_triangle import create_observed_triangle


def predict_and_evaluate():
    # Load saved model and scalers
    mlp = joblib.load("models/mlp_model.pkl")
    input_scaler = joblib.load("models/input_scaler.pkl")
    target_scaler = joblib.load("models/target_scaler.pkl")

    # Load observed triangle from full data
    df = pd.read_csv("data/all_contracts.csv")
    observed_triangle = create_observed_triangle(df, current_year=2025)

    max_dev = 10

    # Prepare test inputs and true ultimate from raw test_df
    test_df = pd.read_csv("data/test_contracts.csv")
    true_ultimate = test_df["ultimate"].values

    X_test = []
    known_paid = []

    for year in test_df["policy_year"]:
        if year not in observed_triangle.index:
            continue
        row = observed_triangle.loc[year].values
        observed_dev = np.count_nonzero(~np.isnan(row))
        known = row[:observed_dev]
        padded_input = list(known) + [0] * (max_dev - observed_dev)
        X_test.append(padded_input)
        known_paid.append(known[-1])

    # Convert to array and scale
    X_test = pd.DataFrame(X_test, columns=[f"dev_{i}" for i in range(max_dev)])
    X_test_scaled = input_scaler.transform(X_test)

    # Predict residuals
    residual_preds_scaled = mlp.predict(X_test_scaled)
    residual_preds = target_scaler.inverse_transform(residual_preds_scaled.reshape(-1, 1)).flatten()

    # Compute final predicted ultimate
    predicted_ultimate = np.array(known_paid) + residual_preds
    true_ultimate = np.array(true_ultimate[:len(predicted_ultimate)])  # ensure same length

    # Evaluate
    abs_error = np.abs(predicted_ultimate - true_ultimate)
    mae = abs_error.mean()
    rmse = np.sqrt(np.mean((predicted_ultimate - true_ultimate) ** 2))

    print(f"Test MAE (Ultimate): {mae:.2f}")
    print(f"Test RMSE (Ultimate): {rmse:.2f}")

    # Save for plotting
    np.save("data/ml_predicted_ultimate.npy", predicted_ultimate)
