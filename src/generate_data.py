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

        # --- Strong disruption ---
        curve = base_curve.copy()

        # Add noisy fluctuations
        bumps = np.random.normal(0, 0.05, size=10)
        curve += bumps

        # Occasionally reverse or scramble
        if np.random.rand() < 0.2:
            curve = np.random.permutation(curve)

        # Blend with reversed base to simulate distorted reporting
        if np.random.rand() < 0.2:
            curve = 0.5 * curve + 0.5 * base_curve[::-1]

        # Inject random shocks
        for _ in range(np.random.randint(2, 5)):
            idx = np.random.randint(1, 9)
            shock = np.random.uniform(-0.2, 0.3)
            curve[idx:] += shock

        # Clip to [0, 1] and re-cumulate
        curve = np.clip(curve, 0, 1)
        curve = np.maximum.accumulate(curve)

        # Add some flat lines or early cutoffs
        if np.random.rand() < 0.2:
            cutoff = np.random.randint(3, 8)
            curve[cutoff:] = curve[cutoff]

        # Set ultimate and scale
        ultimate = np.random.uniform(5000, 50000)
        cumulative_paid = np.round(curve * ultimate).astype(float)
        cumulative_paid[-1] = ultimate  # Ensure final dev = ultimate

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
    print("✅ Saved data/all_contracts.csv with heavy disruption.")
    return df
