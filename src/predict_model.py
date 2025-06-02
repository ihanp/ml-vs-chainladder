import numpy as np
import pandas as pd
import joblib

def predict_and_evaluate():
    # Load saved model and scalers
    mlp = joblib.load("models/mlp_model.pkl")
    input_scaler = joblib.load("models/input_scaler.pkl")
    target_scaler = joblib.load("models/target_scaler.pkl")

    # Load test data
    test_df = pd.read_csv("data/test_contracts.csv")
    max_dev = 9
    CURRENT_YEAR = 2024

    # Prepare test inputs dynamically with 9 features (as used in training)
    X_test = []
    true_ultimate = []
    known_paid = []

    for _, row in test_df.iterrows():
        policy_year = row["policy_year"]
        observed_dev = min(CURRENT_YEAR - policy_year, max_dev)  # max 9 features

        known = [row[f"dev_{i}"] for i in range(observed_dev)]
        padded_input = known + [0] * (max_dev - observed_dev)  # Ensure 9 inputs

        X_test.append(padded_input)
        known_paid.append(known[-1])
        true_ultimate.append(row["ultimate"])

    # Convert to array and scale
    X_test = pd.DataFrame(X_test, columns=[f"dev_{i}" for i in range(max_dev)])
    X_test_scaled = input_scaler.transform(X_test)

    # Predict residuals
    residual_preds_scaled = mlp.predict(X_test_scaled)
    residual_preds = target_scaler.inverse_transform(residual_preds_scaled.reshape(-1, 1)).flatten()

    # Compute final predicted ultimate
    predicted_ultimate = np.array(known_paid) + residual_preds
    true_ultimate = np.array(true_ultimate)

    # Evaluate
    abs_error = np.abs(predicted_ultimate - true_ultimate)
    mae = abs_error.mean()
    rmse = np.sqrt(np.mean((predicted_ultimate - true_ultimate) ** 2))

    print(f"Test MAE (Ultimate): {mae:.2f}")
    print(f"Test RMSE (Ultimate): {rmse:.2f}")

    # Save for plotting
    np.save("data/ml_predicted_ultimate.npy", predicted_ultimate)
