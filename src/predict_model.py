import numpy as np
import pandas as pd
import joblib

def predict_and_evaluate():
    # Load saved model and scalers
    mlp = joblib.load("models/mlp_model.pkl")
    input_scaler = joblib.load("models/input_scaler.pkl")
    target_scaler = joblib.load("models/target_scaler.pkl")

    # Load observed triangle from full data
    df = pd.read_csv("data/all_contracts.csv")
    max_dev = 9

    # Prepare test inputs and true ultimate from raw test_df
    test_df = pd.read_csv("data/test_contracts.csv")
    true_ultimate = test_df["ultimate"].values

    X_test = []
    known_paid = []
    
    CURRENT_YEAR = 2025
    
    for _, row in test_df.iterrows():
        policy_year = int(row["policy_year"])
        observed_dev = max(0, min(CURRENT_YEAR - policy_year + 1, max_dev))  # e.g. 2025 -> 1 dev
    
        devs = []
        cum_paid = 0.0
        for i in range(observed_dev):
            col = f"dev_{i}"
            val = row.get(col, np.nan)
            if pd.notna(val):
                devs.append(val)
                cum_paid += val
            else:
                break  # fallback: stop at first NA
    
        padded_input = devs + [0.0] * (max_dev - len(devs))
        X_test.append(padded_input)
        known_paid.append(cum_paid)

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
    print("Sample known paid values:", known_paid[:5])
    print("Sample scaled input values:", X_test_scaled[:3])
    print("Sample residuals (rescaled):", residual_preds[:3])
    print("Sample predicted ultimate:", predicted_ultimate[:3])
    print("Residual stats:")
    print("Min:", residual_preds.min(), "Max:", residual_preds.max(), "Mean:", residual_preds.mean())

    # Save for plotting
    np.save("data/ml_predicted_ultimate.npy", predicted_ultimate)
