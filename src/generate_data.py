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

        # --- Break chain ladder assumption: devs not multiplicative ---
        # Create non-monotonic or erratic base pattern with random bumps
        bumps = np.random.normal(0, 0.02, size=10)
        custom_curve = base_curve + bumps
        custom_curve = np.clip(custom_curve, 0, 1)
        custom_curve = np.maximum.accumulate(custom_curve)  # still cumulative

        # --- Add disruptive noise to mid-devs ---
        disruption_index = np.random.randint(2, 8)
        custom_curve[disruption_index:] += np.random.normal(0, 0.03)
        custom_curve = np.clip(custom_curve, 0, 1)
        custom_curve = np.maximum.accumulate(custom_curve)

        # --- Force ultimate to match last cumulative ---
        ultimate = np.random.uniform(5000, 50000)
        cumulative_paid = np.round(custom_curve * ultimate)

        # Guarantee last dev equals ultimate
        cumulative_paid[-1] = ultimate

        # Optional plateau or drop
        if np.random.rand() < 0.1:
            cutoff = np.random.randint(4, 9)
            cumulative_paid[cutoff:] = cumulative_paid[cutoff]

        # --- Save row ---
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
