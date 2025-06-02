import numpy as np
import pandas as pd
import os

def generate_synthetic_contracts(n_contracts=100000, seed=42):
    np.random.seed(seed)
    base_curve = np.array([0.25, 0.45, 0.60, 0.70, 0.78, 0.85, 0.90, 0.94, 0.97, 1.00])
    data = []

    for i in range(n_contracts):
        contract_id = f"C{i:05d}"
        policy_year = np.random.randint(2010, 2026)

        # --- Add per-contract shape noise (curve distortion) ---
        shape_shift = np.random.normal(1.0, 0.1)
        curve = base_curve ** shape_shift

        # --- Add multiplicative noise to simulate volatility ---
        noise = np.random.normal(loc=1.0, scale=0.05, size=10)
        dev_pattern = np.maximum.accumulate(curve * noise)
        dev_pattern = np.clip(dev_pattern, 0, 1)

        # --- Create base ultimate ---
        base_ultimate = np.random.uniform(5000, 50000)
        cumulative_paid = np.round(base_ultimate * dev_pattern)

        # --- Optional: simulate plateauing or claim cutoff ---
        if np.random.rand() < 0.05:
            cutoff = np.random.randint(4, 9)
            cumulative_paid[cutoff:] = cumulative_paid[cutoff]

        # --- Set ultimate to final dev (or slightly perturbed) ---
        ultimate = cumulative_paid[-1] + np.random.normal(0, 250)
        ultimate = max(ultimate, cumulative_paid[-1])  # Ensure non-decreasing

        # --- Build row as before ---
        row = {
            "contract_id": contract_id,
            "policy_year": policy_year,
            "ultimate": ultimate
        }
        for dev in range(10):
            row[f"dev_{dev}"] = cumulative_paid[dev]
        data.append(row)

    df = pd.DataFrame(data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/all_contracts.csv", index=False)
    print("Saved data/all_contracts.csv")
    return df
